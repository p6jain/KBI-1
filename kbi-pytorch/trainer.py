import random
import numpy
import time
import evaluate
import torch
import kb
import utils
import os

class Trainer(object):
    def __init__(self, scoring_function, regularizer, loss, optim, train, valid, test, verbose=0, batch_size=1000,
                 hooks=None , eval_batch=100, negative_count=10, gradient_clip=None, regularization_coefficient=0.01,
                 save_dir="./logs", model_name = None, image_compatibility = None, image_compatibility_coefficient = 0.01):
        super(Trainer, self).__init__()
        self.model_name = model_name

        self.scoring_function = scoring_function
        self.loss = loss
        self.regularizer = regularizer
        self.image_compatibility = image_compatibility
        self.image_compatibility_coefficient = image_compatibility_coefficient

        self.train = train
        self.test = test
        self.valid = valid
        self.optim = optim
        self.batch_size = batch_size
        self.negative_count = negative_count
        self.ranker = evaluate.ranker(self.scoring_function, kb.union([train.kb, valid.kb, test.kb]))
        self.eval_batch = eval_batch
        self.gradient_clip = gradient_clip
        self.regularization_coefficient = regularization_coefficient
        self.save_directory = save_dir
        self.best_mrr_on_valid = {"valid_m":{"mrr":0.0}, "test_m":{"mrr":0.0},
                                          "valid_e2":{"mrr":0.0}, "test_e2":{"mrr":0.0},
                                          "valid_e1":{"mrr":0.0}, "test_e1":{"mrr":0.0}}#{"valid":{"mrr":0.0}}
        self.verbose = verbose
        self.hooks = hooks if hooks else []

    def step_neg_sens_2step(self):
        s, r, o, ns, no, ns2, no2 = self.train.tensor_sample(self.batch_size, self.negative_count)
        fp = self.scoring_function(s, r, o)

        #only type incorrect samples
        fns = self.scoring_function(ns, r, o)
        fno = self.scoring_function(s, r, no)

        if self.regularization_coefficient is not None:
            reg = self.regularizer(s, r, o) + self.regularizer(ns, r, o) + self.regularizer(s, r, no)
            reg = reg/(self.batch_size*self.scoring_function.embedding_dim*(1+2*self.negative_count))
        else:
            reg = 0
        loss = self.loss(fp, fns, fno) + self.regularization_coefficient*reg
        x = loss.item()
        rg = reg.item()
        self.optim.zero_grad()
        loss.backward(retain_graph=True)
        if(self.gradient_clip is not None):
            torch.nn.utils.clip_grad_norm(self.scoring_function.parameters(), self.gradient_clip)
        self.optim.step()
        debug = ""
        if "post_epoch" in dir(self.scoring_function):
            debug = self.scoring_function.post_epoch()


        #only type correct samples ---- neg_count/10
        fns2 = self.scoring_function(ns2, r, o)
        fno2 = self.scoring_function(s, r, no2)

        if self.regularization_coefficient is not None:
            reg = self.regularizer(s, r, o) + self.regularizer(ns2, r, o) + self.regularizer(s, r, no2)
            reg = reg/(self.batch_size*self.scoring_function.embedding_dim*(1+2*(self.negative_count/10)))
        else:
            reg = 0
        loss = self.loss(fp, fns2, fno2) + self.regularization_coefficient*reg
        x += loss.item()
        rg += reg.item()
        self.optim.zero_grad()
        loss.backward()
        if(self.gradient_clip is not None):
            torch.nn.utils.clip_grad_norm(self.scoring_function.parameters(), self.gradient_clip)
        self.scoring_function.E_t.weight.grad.zero_()
        self.scoring_function.R_ht.weight.grad.zero_()
        self.scoring_function.R_tt.weight.grad.zero_()
        self.optim.step()
        # debug = ""
        if "post_epoch" in dir(self.scoring_function):
            debug += self.scoring_function.post_epoch()

        return x, rg, debug

    def step(self):
        s, r, o, ns, no = self.train.tensor_sample(self.batch_size, self.negative_count)

        flag = random.randint(1,10001)
        if flag>9600:
            flag_debug = 1
        else:
            flag_debug = 0

        fp = self.scoring_function(s, r, o, flag_debug=flag_debug)
        if flag_debug:
            fns = self.scoring_function(ns, r, o, flag_debug=flag_debug+1)
            fno = self.scoring_function(s, r, no, flag_debug=flag_debug+1)
        else:
            fns = self.scoring_function(ns, r, o, flag_debug=0)
            fno = self.scoring_function(s, r, no, flag_debug=0)
        if self.regularization_coefficient is not None:
            reg = self.regularizer(s, r, o) + self.regularizer(ns, r, o) + self.regularizer(s, r, no)
            reg = reg/(self.batch_size*self.scoring_function.embedding_dim*(1+2*self.negative_count))
        else:
            reg = 0

        loss = self.loss(fp, fns, fno) + self.regularization_coefficient*reg

        x = loss.item()
        rg = reg.item()
        self.optim.zero_grad()
        loss.backward()
        if(self.gradient_clip is not None):
            torch.nn.utils.clip_grad_norm(self.scoring_function.parameters(), self.gradient_clip)
        self.optim.step()
        debug = ""
        if "post_epoch" in dir(self.scoring_function):
            debug = self.scoring_function.post_epoch()
        return x, rg, debug



    def step_icml(self):
        s, r, o, ns, no = self.train.tensor_sample(self.batch_size, self.negative_count)

        flag = random.randint(1,10001)
        if flag>9600:
            flag_debug = 1
        else:
            flag_debug = 0
        fp = self.scoring_function(s, r, o, flag_debug=flag_debug)
        if flag_debug:
            fns = self.scoring_function(ns, r, o, flag_debug=flag_debug+1)
            fno = self.scoring_function(s, r, no, flag_debug=flag_debug+1)
        else:
            fns = self.scoring_function(ns, r, o, flag_debug=0)
            fno = self.scoring_function(s, r, no, flag_debug=0)
        if self.regularization_coefficient is not None:
            reg = self.regularizer(s, r, o) + self.regularizer(ns, r, o) + self.regularizer(s, r, no)
            reg = reg/(self.batch_size*self.scoring_function.embedding_dim*(1+2*self.negative_count))
        else:
            reg = 0

        loss = self.loss(fp, fns, fno) + self.regularization_coefficient*reg

        num_relations = len(self.train.kb.relation_map)
        r_rev = (r + num_relations)%(2*num_relations) 
        s_rev, o_rev, ns_rev, no_rev = o, s, no, ns

        fp_rev = self.scoring_function(s_rev, r_rev, o_rev, flag_debug=flag_debug)
        if flag_debug:
            fns_rev = self.scoring_function(ns_rev, r_rev, o_rev, flag_debug=flag_debug+1)
            fno_rev = self.scoring_function(s_rev, r_rev, no_rev, flag_debug=flag_debug+1)
        else:
            fns_rev = self.scoring_function(ns_rev, r_rev, o_rev, flag_debug=0)
            fno_rev = self.scoring_function(s_rev, r_rev, no_rev, flag_debug=0)
        if self.regularization_coefficient is not None:
            reg_rev = self.regularizer(s_rev, r_rev, o_rev) + self.regularizer(ns_rev, r_rev, o_rev) + self.regularizer(s_rev, r_rev, no_rev)
            reg_rev = reg/(self.batch_size*self.scoring_function.embedding_dim*(1+2*self.negative_count))
        else:
            reg_rev = 0
    
        loss += self.loss(fp_rev, fns_rev, fno_rev) + self.regularization_coefficient*reg_rev

        x = loss.item()
        rg = reg.item()
        self.optim.zero_grad()
        loss.backward()
        if(self.gradient_clip is not None):
            torch.nn.utils.clip_grad_norm(self.scoring_function.parameters(), self.gradient_clip)
        self.optim.step()
        debug = ""
        if "post_epoch" in dir(self.scoring_function):
            debug = self.scoring_function.post_epoch()
        return x, rg, debug

    def step_aux(self):
        #s, r, o, ns, no, s_image, o_image = self.train.tensor_sample(self.batch_size, self.negative_count)
        s, r, o, ns, no = self.train.tensor_sample(self.batch_size, self.negative_count)

        flag = random.randint(1,10001)
        if flag>9600:
            flag_debug = 1
        else:
            flag_debug = 0

        fp = self.scoring_function(s, r, o, flag_debug=flag_debug)
        if flag_debug:
            fns = self.scoring_function(ns, r, o, flag_debug=flag_debug+1)
            fno = self.scoring_function(s, r, no, flag_debug=flag_debug+1)
        else:
            fns = self.scoring_function(ns, r, o, flag_debug=0)
            fno = self.scoring_function(s, r, no, flag_debug=0)
        if self.regularization_coefficient is not None:
            reg = self.regularizer(s, r, o) + self.regularizer(ns, r, o) + self.regularizer(s, r, no)
            reg = reg/(self.batch_size*self.scoring_function.embedding_dim*(1+2*self.negative_count))
        else:
            reg = 0

        ic_score_s, ic_score_o = self.image_compatibility(s, o)#, s_image, o_image)

        #print("Prachi Debug", "ic_score_s",ic_score_s.shape)
        #print("Prachi Debug", "reg", reg.shape, reg)
        CGREEN  = '\33[32m';CRED = '\033[91m';CVIOLET = '\33[35m'
        CEND = '\033[0m'
        tmp=self.loss(fp, fns, fno)
        print("Prachi Debug", "self.loss(fp, fns, fno)", tmp.shape, CGREEN, tmp, CEND)
        print("Prachi Debug","ic_score_s","ic_score_o",CRED, ic_score_s,ic_score_o,CEND)
        image_compatibility_loss = torch.mean(torch.stack((ic_score_s,ic_score_o))).squeeze()
        print("Prachi Debug","image_compatibility_loss",image_compatibility_loss.shape,CVIOLET,image_compatibility_loss, CEND)
        #print("Prachi Debug::", self.image_compatibility_coefficient)
        loss = self.loss(fp, fns, fno) + self.regularization_coefficient*reg + self.image_compatibility_coefficient*image_compatibility_loss#(ic_score_s+ic_score_o)

        x = loss.item()
        rg = reg.item()
        self.optim.zero_grad()
        loss.backward()
        if(self.gradient_clip is not None):
            torch.nn.utils.clip_grad_norm(self.scoring_function.parameters(), self.gradient_clip)
        self.optim.step()##check
        debug = ""
        if "post_epoch" in dir(self.scoring_function):
            debug = self.scoring_function.post_epoch()
        return x, rg, debug

    def save_state(self, mini_batches, valid_score, test_score):
        state = dict()
        state['mini_batches'] = mini_batches
        state['epoch'] = mini_batches*self.batch_size/self.train.kb.facts.shape[0]
        state['model_name'] = type(self.scoring_function).__name__
        state['model_weights'] = self.scoring_function.state_dict()
        state['optimizer_state'] = self.optim.state_dict()
        state['optimizer_name'] = type(self.optim).__name__
        state['valid_score_e2'] = valid_score['e2']
        state['test_score_e2'] = test_score['e2']
        state['valid_score_e1'] = valid_score['e1']
        state['test_score_e1'] = test_score['e1']
        state['valid_score_m'] = valid_score['m']
        state['test_score_m'] = test_score['m']

        state['entity_map'] = self.train.kb.entity_map
        state['reverse_entity_map'] = self.train.kb.reverse_entity_map
        state['type_entity_range'] = self.train.kb.type_entity_range

        filename = os.path.join(self.save_directory,
                                    "epoch_%.1f_val_%5.2f_%5.2f_%5.2f_test_%5.2f_%5.2f_%5.2f.pt"%(state['epoch'],
                                           state['valid_score_e2']['mrr'],
                                           state['valid_score_e1']['mrr'],
                                           state['valid_score_m']['mrr'],
                                           state['test_score_e2']['mrr'],
                                           state['test_score_e1']['mrr'],
                                           state['test_score_m']['mrr']))


        #torch.save(state, filename)
        try:
            if(state['valid_score_m']['mrr'] >= self.best_mrr_on_valid["valid_m"]["mrr"]):
                print("Best Model details:\n","valid_m",str(state['valid_score_m']), "test_m",str(state["test_score_m"]),
                                          "valid_e2",str(state['valid_score_e2']), "test_e2",str(state["test_score_e2"]),
                                          "valid_e1",str(state['valid_score_e1']),"test_e1",str(state["test_score_e1"]))
                best_name = os.path.join(self.save_directory, "best_valid_model.pt")
                self.best_mrr_on_valid = {"valid_m":state['valid_score_m'], "test_m":state["test_score_m"],
                                          "valid_e2":state['valid_score_e2'], "test_e2":state["test_score_e2"],
                                          "valid_e1":state['valid_score_e1'], "test_e1":state["test_score_e1"]}

                if(os.path.exists(best_name)):
                    os.remove(best_name)
                torch.save(state, best_name)#os.symlink(os.path.realpath(filename), best_name)
        except:
            utils.colored_print("red", "unable to save model")

    def load_state(self, state_file):
        state = torch.load(state_file)
        if state['model_name'] != type(self.scoring_function).__name__:
            utils.colored_print('yellow', 'model name in saved file %s is different from the name of current model %s' %
                                (state['model_name'], type(self.scoring_function).__name__))
        self.scoring_function.load_state_dict(state['model_weights'])
        if state['optimizer_name'] != type(self.optim).__name__:
            utils.colored_print('yellow', ('optimizer name in saved file %s is different from the name of current '+
                                          'optimizer %s') %
                                (state['optimizer_name'], type(self.optim).__name__))
        self.optim.load_state_dict(state['optimizer_state'])
        return state['mini_batches']

    def start(self, steps=50, batch_count=(20, 10), mb_start=0):
        start = time.time()
        losses = []
        count = 0;
        if self.model_name == 'image_model':
            step_fn = self.step_aux
        elif self.train.flag_add_reverse:
            step_fn = self.step_icml
        else:
            step_fn = self.step
        for i in range(mb_start, steps):
            l, reg, debug = step_fn()
            losses.append(l)
            suffix = ("| Current Loss %8.4f | "%l) if len(losses) != batch_count[0] else "| Average Loss %8.4f | " % \
                                                                                         (numpy.mean(losses))
            suffix += "reg %6.3f | time %6.0f ||"%(reg, time.time()-start)
            suffix += debug
            prefix = "Mini Batches %5d or %5.1f epochs"%(i+1, i*self.batch_size/self.train.kb.facts.shape[0])
            utils.print_progress_bar(len(losses), batch_count[0],prefix=prefix, suffix=suffix)
            if len(losses) >= batch_count[0]:
                losses = []
                count += 1
                if count == batch_count[1]:
                    self.scoring_function.eval()
                    valid_score = evaluate.evaluate("valid", self.ranker, self.valid.kb, self.eval_batch,
                                                    verbose=self.verbose, hooks=self.hooks)
                    test_score = evaluate.evaluate("test ", self.ranker, self.test.kb, self.eval_batch,
                                                   verbose=self.verbose, hooks=self.hooks)
                    self.scoring_function.train()
                    count = 0
                    print()
                    self.save_state(i, valid_score, test_score)
        print()
        print("Ending")
        print(self.best_mrr_on_valid["valid_m"])
        print(self.best_mrr_on_valid["test_m"])
