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
                 save_dir="./logs", model_name = None, image_compatibility = None, image_compatibility_coefficient = 0.01, scheduler=None):
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
        self.ranker = evaluate.ranker(self.scoring_function, kb.union([train.kb, valid.kb, test.kb]), num_relations = len(train.kb.relation_map))
        self.eval_batch = eval_batch
        self.gradient_clip = gradient_clip
        self.regularization_coefficient = regularization_coefficient
        self.save_directory = save_dir
        self.best_mrr_on_valid = {"valid_m":{"mrr":0.0}, "test_m":{"mrr":0.0},
                                          "valid_e2":{"mrr":0.0}, "test_e2":{"mrr":0.0},
                                          "valid_e1":{"mrr":0.0}, "test_e1":{"mrr":0.0}}#{"valid":{"mrr":0.0}}
        self.verbose = verbose
        self.hooks = hooks if hooks else []

        self.scheduler = scheduler

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

    def step(self):#_lacroix(self):
        #s, r, o, ns, no, _, _, _ = self.train.tensor_sample(self.batch_size, self.negative_count)
        s, r, o, ns, no = self.train.tensor_sample(self.batch_size, self.negative_count)

        flag = random.randint(1,10001)
        if flag>9950:
            flag_debug = 1
        else:
            flag_debug = 0

        if flag_debug:
            scores_e1 = self.scoring_function(s, r, no, flag_debug=flag_debug+1)
            scores_e2 = self.scoring_function(ns, r, o, flag_debug=flag_debug+1)
        else:
            scores_e1 = self.scoring_function(s, r, no, flag_debug=0)
            scores_e2 = self.scoring_function(ns, r, o, flag_debug=0)

        if self.regularization_coefficient is not None:
            reg = self.regularizer(s, r, o) #+ self.regularizer(None, r, o) + self.regularizer(s, r, None)
            #reg = self.regularizer(s, r, o) + self.regularizer(ns, r, o) + self.regularizer(s, r, no)
            ##
            #Issue for typed models demom.. embedding dim = type model embedding dim << base model embedding dim (embeddings mostly used for reg )
            #Consider using a smaller (10 times) reg penalty in type model than what was used for complex
            reg = reg/(self.batch_size*self.scoring_function.embedding_dim)#*(1+2*self.negative_count))
            
            ##
            #reg = self.regularizer(s, r, o) + self.regularizer(ns, r, o) + self.regularizer(s, r, no) + self.regularizer(ns, r, no)
        else:
            reg = 0


        if self.loss.name == "crossentropy_loss":
            loss = self.loss(scores_e1, o) + self.loss(scores_e2, s) + self.regularization_coefficient*reg
        else:
            loss = self.loss(fp, scores_e2, scores_e1) + self.regularization_coefficient*reg


        #loss = self.loss(scores_e1, o) + self.loss(scores_e2, s) + self.regularization_coefficient*reg

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


    def step_back(self):
        #s, r, o, ns, no, _, _, _ = self.train.tensor_sample(self.batch_size, self.negative_count)
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

        if self.loss.name == "crossentropy_loss":
            loss = self.loss(fno, o) + self.loss(fns, s) + self.regularization_coefficient*reg
        else:
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
        #s, r, o, ns, no = self.train.tensor_sample(self.batch_size, self.negative_count)
        s, r, o, ns, no, s_prob, r_prob, o_prob, ns_prob, no_prob = self.train.tensor_sample(self.batch_size, self.negative_count)

        flag = random.randint(1,10001)
        if flag>9600:
            flag_debug = 1
        else:
            flag_debug = 0
        fp = self.scoring_function(s, r, o, flag_debug=flag_debug)
        if flag_debug:
            #fns = self.scoring_function(ns, r, o, flag_debug=flag_debug+1)
            fno = self.scoring_function(s, r, no, flag_debug=flag_debug+1)
        else:
            #fns = self.scoring_function(ns, r, o, flag_debug=0)
            fno = self.scoring_function(s, r, no, flag_debug=0)

        
        if 1:#self.regularization_coefficient is not None:
            #s_prob = s_prob.type(torch.cuda.FloatTensor).unsqueeze(-1)
            #o_prob = o_prob.type(torch.cuda.FloatTensor).unsqueeze(-1)
            #r_prob = r_prob.type(torch.cuda.FloatTensor).unsqueeze(-1)
            #ns_prob = ns_prob.type(torch.cuda.FloatTensor).unsqueeze(-1)
            #no_prob = no_prob.type(torch.cuda.FloatTensor).unsqueeze(-1)
            '''
            print("Prachi Debug","s",s.shape,s_prob.shape)
            print("Prachi Debug","r",r.shape,r_prob.shape)
            print("Prachi Debug","o",o.shape,o_prob.shape)
            print("Prachi Debug","ns",ns.shape,ns_prob.shape)
            print("Prachi Debug","no",no.shape,no_prob.shape)'''
            #reg = self.regularizer(s, r, o, s_prob, r_prob, o_prob) + self.regularizer(ns, r, o, ns_prob, r_prob, o_prob) + self.regularizer(s, r, no, s_prob, r_prob, no_prob)
            reg = self.regularizer(s, r, o) #+ self.regularizer(s, r, no)#self.regularizer(ns, r, o) + self.regularizer(s, r, no)
            #reg = reg/(self.batch_size*self.scoring_function.embedding_dim*(1+2*self.negative_count))
            if no is None:
                reg = reg/(self.batch_size*self.scoring_function.embedding_dim)#*(1 + 1*self.train.num_entity))
            else:
                reg = reg/(self.batch_size*self.scoring_function.embedding_dim*(1 + 1*self.negative_count))
        else:
            reg = 0#torch.tensor([0])
        
        if self.loss.name == "crossentropy_loss":
            loss = self.loss(fno, o) + self.regularization_coefficient*reg #self.loss(fns, s) + self.regularization_coefficient*reg
        else:
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

    def step_aux(self):
        #s, r, o, ns, no, s_image, o_image = self.train.tensor_sample(self.batch_size, self.negative_count)
        #s, r, o, ns, no = self.train.tensor_sample(self.batch_size, self.negative_count)
        s, r, o, ns, no, s_im, o_im, ns_im, no_im, s_oov, o_oov, ic_r_regC = self.train.tensor_sample(self.batch_size, self.negative_count)

        flag = random.randint(1,10001)
        if flag>9600:
            flag_debug = 1
        else:
            flag_debug = 0

        fp = self.scoring_function(s, r, o, s_im, o_im, flag_debug=flag_debug)
        if flag_debug:
            fns = self.scoring_function(ns, r, o, ns_im, o_im, flag_debug=flag_debug+1)
            fno = self.scoring_function(s, r, no, s_im, no_im, flag_debug=flag_debug+1)
        else:
            fns = self.scoring_function(ns, r, o, ns_im, o_im, flag_debug=0)
            fno = self.scoring_function(s, r, no, s_im, no_im, flag_debug=0)
        if self.regularization_coefficient is not None:
            reg = self.regularizer(s, r, o) + self.regularizer(ns, r, o) + self.regularizer(s, r, no)
            reg = reg/(self.batch_size*self.scoring_function.embedding_dim*(1+2*self.negative_count))
        else:
            reg = 0

        #ic_score_s, ic_score_o ,tmp_ic_score_s_r, tmp_ic_score_s_o= self.image_compatibility(s,r, o)#, s_image, o_image)
        ic_score_s, ic_score_o, im_score  = self.image_compatibility(s,r, o, s_im, o_im)#, s_image, o_image)


        #print("Prachi Debug", "ic_score_s",ic_score_s.shape)
        #print("Prachi Debug", "reg", reg.shape, reg)
        #print("Prachi Debug", "im_score",im_score.shape)
        verbose_debug = random.randint(1,10001)
        if verbose_debug > 9999:
            CGREEN  = '\33[32m';CRED = '\033[91m';CVIOLET = '\33[35m'
            CEND = '\033[0m'
            tmp=self.loss(fp, fns, fno)
            print("Prachi Debug", "self.loss(fp, fns, fno)", tmp.shape, CGREEN, tmp, CEND)
            print("Prachi Debug","ic_score_s","ic_score_o",CRED, torch.mean(ic_score_s),torch.mean(ic_score_o), torch.mean(im_score),CEND)

        if self.image_compatibility_coefficient:
            ##
            s_oov = s_oov.type(torch.cuda.FloatTensor)
            o_oov = o_oov.type(torch.cuda.FloatTensor)
            ic_score_s = ic_score_s.type(torch.cuda.FloatTensor)
            ic_score_o = ic_score_o.type(torch.cuda.FloatTensor)
            #print("Prachi Debug", "s_oov",s_oov.shape)
            #print("Prachi Debug", "ic_score_s", ic_score_s.type())
            ic_score_s = s_oov * ic_score_s
            ic_score_o = o_oov * ic_score_o
            ic_r_regC = ic_r_regC.type(torch.cuda.FloatTensor)
            im_score = ic_r_regC * im_score
            ##

            image_compatibility_loss = torch.mean(torch.stack((ic_score_s,ic_score_o))).squeeze()
            im_score_loss = torch.mean(im_score)
            #print("Prachi Debug", "image_compatibility_loss",image_compatibility_loss.shape)
            #print("Prachi Debug", "im_score_loss",im_score_loss.shape)
            #image_compatibility_loss = torch.mean(torch.stack((ic_score_s,ic_score_o,tmp_ic_score_s_r, tmp_ic_score_s_o))).squeeze()
            if verbose_debug > 9999:
                print("Prachi Debug","image_compatibility_loss",CVIOLET,image_compatibility_loss, CEND)
                print("Prachi Debug","im_score_loss",CVIOLET,im_score_loss, CEND)
            #print("Prachi Debug::", self.image_compatibility_coefficient)
        else:
            image_compatibility_loss = 0.0
            im_score_loss = 0.0
        loss = self.loss(fp, fns, fno) + self.regularization_coefficient*reg + self.image_compatibility_coefficient*(image_compatibility_loss + im_score_loss)#(ic_score_s+ic_score_o)

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
        #state['type_entity_range'] = self.train.kb.type_entity_range
        state['relation_map'] = self.train.kb.relation_map
        state['reverse_relation_map'] = self.train.kb.reverse_relation_map 

        state['im_entity_map'] = self.train.kb.im_entity_map
        state['im_reverse_entity_map'] = self.train.kb.im_reverse_entity_map 
        state['mid_imid_map'] = self.train.kb.mid_imid_map
        state['additional_params'] = self.train.kb.additional_params
        state['nonoov_entity_count'] = self.train.kb.nonoov_entity_count 


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
        print("Loading!", state_file, state['model_name'])
        if state['model_name'] != type(self.scoring_function).__name__:
            utils.colored_print('yellow', 'model name in saved file %s is different from the name of current model %s' %
                                (state['model_name'], type(self.scoring_function).__name__))

        if 1:#try:
            if self.scoring_function._get_name() == state['model_name']:
                self.scoring_function.load_state_dict(state['model_weights'])
            elif self.scoring_function.base_model._get_name() == state['model_name']:
                self.scoring_function.base_model.load_state_dict(state['model_weights'])  
            elif self.scoring_function._get_name().split("_v1")[0] == state['model_name']:
                self.scoring_function.load_state_dict(state['model_weights'], strict=False) 
        #except:
        #    print("Wrong model file loaded!", self.scoring_function._get_name(), state['model_name'])

        if state['optimizer_name'] != type(self.optim).__name__:
            utils.colored_print('yellow', ('optimizer name in saved file %s is different from the name of current '+
                                          'optimizer %s') %
                                (state['optimizer_name'], type(self.optim).__name__))

        if self.scoring_function._get_name() == state['model_name']:
            self.optim.load_state_dict(state['optimizer_state'])

        return state['mini_batches']

    def start(self, steps=50, batch_count=(20, 10), mb_start=0):
        start = time.time()
        losses = []
        count = 0;
        if self.model_name == "image_model" or self.model_name == "only_image_model" or self.model_name == "typed_image_model" or self.model_name == "typed_image_model_reg":#self.model_name == 'image_model' or self.model_name == 'typed_image_model_reg':
            step_fn = self.step_aux
        elif self.train.flag_add_reverse:
            step_fn = self.step_icml
        else:
            step_fn = self.step
        '''
        self.scoring_function.eval()
        test_score = evaluate.evaluate("test ", self.ranker, self.test.kb, self.eval_batch,
                                                   verbose=self.verbose, hooks=self.hooks)
        valid_score = evaluate.evaluate("valid", self.ranker, self.valid.kb, self.eval_batch,
                                                    verbose=self.verbose, hooks=self.hooks)
        self.scoring_function.train()
        '''
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

                #beta = torch.nn.Sigmoid()(self.scoring_function.beta.weight.data)
                #print("Prachi Debug", "beta", beta[:50], beta.sum(), beta.mean(), beta.std())
                if count == batch_count[1]:
                    self.scoring_function.eval()
                    valid_score = evaluate.evaluate("valid", self.ranker, self.valid.kb, self.eval_batch,
                                                    verbose=self.verbose, hooks=self.hooks, flag_add_reverse=self.train.flag_add_reverse)
                    test_score = evaluate.evaluate("test ", self.ranker, self.test.kb, self.eval_batch,
                                                   verbose=self.verbose, hooks=self.hooks, flag_add_reverse=self.train.flag_add_reverse)
                    self.scoring_function.train()
                    self.scheduler.step(valid_score['m']['mrr']) #Scheduler to manage learning rate added
                    count = 0
                    print()
                    self.save_state(i, valid_score, test_score)
        print()
        print("Ending")
        print(self.best_mrr_on_valid["valid_m"])
        print(self.best_mrr_on_valid["test_m"])
