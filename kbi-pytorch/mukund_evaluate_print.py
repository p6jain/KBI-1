import utils
import numpy
import torch
import time
import gc
import pprint
import csv

class ranker(object):
    """
    A network that ranks entities based on a scoring function. It excludes entities that have already
    been seen in any kb to compute the ranking as in ####### cite paper here ########. It can be constructed
    from any scoring function/model from models.py
    """
    def __init__(self, scoring_function, all_kb):
        """
        The initializer\n
        :param scoring_function: The model function to used to rank the entities
        :param all_kb: A union of all the knowledge bases (train/test/valid)
        """
        super(ranker, self).__init__()
        self.scoring_function = scoring_function
        self.all_kb = all_kb
        self.knowns = {}
        self.knowns_e1 = {}
        print("building all known database from joint kb")
        for fact in self.all_kb.facts:
            if (fact[0], fact[1]) not in self.knowns:
                self.knowns[(fact[0], fact[1])] = set()
            self.knowns[(fact[0], fact[1])].add(fact[2])
            if (fact[1], fact[2]) not in self.knowns_e1:
                self.knowns_e1[(fact[1], fact[2])] = set()
            self.knowns_e1[(fact[1], fact[2])].add(fact[0])
        print("converting to lists")
        for k in self.knowns:
            self.knowns[k] = list(self.knowns[k])
        for k in self.knowns_e1:
            self.knowns_e1[k] = list(self.knowns_e1[k])
        print("done")

    def get_knowns(self, s, r, flag_e):
        """
        computes and returns the set of all entites that have been seen as a fact (s, r, _)\n
        :param s: The head of the fact
        :param r: The relation of the fact
        :return: All entites o such that (s, r, o) has been seen in all_kb
        """
        if flag_e == "2":
            ks = [self.knowns[(a, b)] for a,b in zip(s, r)]
        elif flag_e == "1":
            ks = [self.knowns_e1[(a, b)] for a,b in zip(s, r)]
        lens = [len(x) for x in ks]
        max_lens = max(lens)
        ks = [numpy.pad(x, (0, max_lens-len(x)), 'edge') for x in ks]
        result= numpy.array(ks)
        return result

    def forward(self, s, r, o, knowns, entity_type_array, flag_eval=2):
        """
        Returns the rank of o for the query (s, r, _) as given by the scoring function\n
        :param s: The head of the query
        :param r: The relation of the query
        :param o: The Gold object of the query
        :param knowns: The set of all o that have been seen in all_kb with (s, r, _) as given by ket_knowns above
        :param flag_eval: Value 2 is used to evaluate <e1,r,?> and value 1 is used to eval <?,r,e2>
        :return: rank of o, score of each entity and score of the gold o
        """
        if flag_eval == 1:
            # scores  = self.scoring_function(None, r, o).data     
            scores, base, head, tail  = self.scoring_function(None, r, o)     
            
            scores = scores.data
            base = base.data
            head = head.data
            tail = tail.data
            
            score_of_expected = scores.gather(1, s.data)
            scores.scatter_(1, o.data, self.scoring_function.minimum_value)
            """
            print("?ro")
            print(s.data)
            print(entity_type_array)
            print(entity_type_array.shape)
            print(entity_type_array[s.data[0]])
            print(numpy.where(entity_type_array[s.data[0]] != entity_type_array))
            print(torch.from_numpy(numpy.where(entity_type_array[s.data[0]] != entity_type_array)[1]))
            print(numpy.where(entity_type_array[s.data[0]] != entity_type_array)[1].shape)
            print(torch.from_numpy(numpy.where(entity_type_array[s.data[0]] != entity_type_array)[1]).shape)
            """
            # scores.scatter_(1, torch.from_numpy(numpy.where(entity_type_array[s.data[0]] != entity_type_array)[0]).cuda().unsqueeze(0), self.scoring_function.minimum_value) # to consider entities of same type only
            """
            head.scatter_(1, o.data, self.scoring_function.minimum_value)
            base.scatter_(1, o.data, self.scoring_function.minimum_value)
            
            base_exp = base.gather(1, s.data)
            head_exp = head.gather(1, s.data)
            # tail_exp = tail.gather(1, s.data)

            head.scatter_(1, knowns, self.scoring_function.minimum_value)

            type_exp = head_exp
            type_tot = head
            """

        else:
            # scores  = self.scoring_function(s, r, None).data
            scores, base, head, tail  = self.scoring_function(s, r, None)
            # scores = scores.data
            # base = base.data.cpu().numpy()[0][0]
            # head = head.data.cpu().numpy()[0][0]
            # tail = tail.data.cpu().numpy()[0][0]
            # score_of_expected = scores.gather(1, o.data)
            
            scores = scores.data
            base = base.data
            head = head.data
            tail = tail.data
            
            score_of_expected = scores.gather(1, o.data)
            scores.scatter_(1, s.data, self.scoring_function.minimum_value)
            """
            print("sr?")
            print(o.data)
            print(entity_type_array)
            print(entity_type_array.shape)
            print(entity_type_array[o.data[0]])
            print(numpy.where(entity_type_array[o.data[0]] != entity_type_array))
            print(torch.from_numpy(numpy.where(entity_type_array[o.data[0]] != entity_type_array)[1]).unsqueeze(0))
            print(numpy.where(entity_type_array[o.data[0]] != entity_type_array)[1].shape)
            print(torch.from_numpy(numpy.where(entity_type_array[o.data[0]] != entity_type_array)[1]).unsqueeze(0).shape)
            """
            # scores.scatter_(1, torch.from_numpy(numpy.where(entity_type_array[o.data[0]] != entity_type_array)[0]).cuda().unsqueeze(0) , self.scoring_function.minimum_value) # to consider entities of same type only

            """
            tail.scatter_(1, s.data, self.scoring_function.minimum_value)
            base.scatter_(1, s.data, self.scoring_function.minimum_value)
            
            base_exp = base.gather(1, o.data)
            # head_exp = head.gather(1, o.data)
            tail_exp = tail.gather(1, o.data)

            tail.scatter_(1, knowns, self.scoring_function.minimum_value)

            type_exp = tail_exp
            type_tot = tail
            """
        
        scores.scatter_(1, knowns, self.scoring_function.minimum_value)
        
        #### 2 step ranking
        """
        base.scatter_(1, knowns, self.scoring_function.minimum_value)
        type_exp = torch.round(type_exp*5.0)
        type_class = torch.round(type_tot*5.0)
        greater = type_class.gt(type_exp).float()
        rank = greater.sum(dim=1)

        out_of_class = type_class.eq(type_exp).float()
        base = base*out_of_class
        greater_base = base.gt(base_exp).float()
        eq_base = base.eq(base_exp).float()
        rank = rank + greater_base.sum(dim=1) + 1 + eq_base.sum(dim=1)/2.0
        return rank, scores, score_of_expected, base, base_exp
        """
        #####Original
        # """
        # print(scores)
        greater = scores.ge(score_of_expected).float()
        equal = scores.eq(score_of_expected).float()
        rank = greater.sum(dim=1)+1+equal.sum(dim=1)/2.0
        return rank, scores, score_of_expected
        # """


def evaluate(name, ranker, kb, batch_size, verbose=0, top_count=5, hooks=None):
    """
    Evaluates an entity ranker on a knowledge base, by computing mean reverse rank, mean rank, hits 10 etc\n
    Can also print type prediction score with higher verbosity.\n
    :param name: A name that is displayed with this evaluation on the terminal
    :param ranker: The ranker that is used to rank the entites
    :param kb: The knowledge base to evaluate on. Must be augmented with type information when used with higher verbosity
    :param batch_size: The batch size of each minibatch
    :param verbose: The verbosity level. More info is displayed with higher verbosity
    :param top_count: The number of entities whose details are stored
    :param hooks: The additional hooks that need to be run with each mini-batch
    :return: A dict with the mrr, mr, hits10 and hits1 of the ranker on kb
    """
    if hooks is None:
        hooks = []
    totals = {"e2":{"mrr":0, "mr":0, "hits10":0, "hits1":0}, "e1":{"mrr":0, "mr":0, "hits10":0, "hits1":0}, "m":{"mrr":0, "mr":0, "hits10":0, "hits1":0}}
    start_time = time.time()
    facts = kb.facts
    if(verbose>0):
        totals["e1"]["correct_type"] = 0
        totals["e2"]["correct_type"] = 0
        entity_type_matrix = kb.entity_type_matrix.cuda()
        
        entity_name_map = kb.entity_name_map
        type_name_map = kb.type_name_map
        reverse_relation_map = kb.reverse_relation_map
        reverse_entity_map = kb.reverse_entity_map

        entity_type_array = kb.entity_type_matrix_np
        
        for hook in hooks:
            hook.begin()
    all_rows = []
    subject_type_total = {}
    subject_type_correct = {}
    object_type_total = {}
    object_type_correct = {}
    relation_subject_type_total = {}
    relation_subject_type_correct = {}
    relation_object_type_total = {}
    relation_object_type_correct = {}
    abs = []
    bds = []
    for i in range(0, int(facts.shape[0]), batch_size):
        start = i
        end = min(i+batch_size, facts.shape[0])
        s = facts[start:end, 0]
        r = facts[start:end, 1]
        o = facts[start:end, 2]
        
        s2 = facts[start:end, 0]
        r2 = facts[start:end, 1]
        o2 = facts[start:end, 2]


        knowns    = ranker.get_knowns(s, r, "2")
        knowns_e1 = ranker.get_knowns(r, o, "1")
        s = torch.autograd.Variable(torch.from_numpy(s).cuda().unsqueeze(1), requires_grad=False)
        r = torch.autograd.Variable(torch.from_numpy(r).cuda().unsqueeze(1), requires_grad=False)
        o = torch.autograd.Variable(torch.from_numpy(o).cuda().unsqueeze(1), requires_grad=False)
        knowns = torch.from_numpy(knowns).cuda()
        knowns_e1 = torch.from_numpy(knowns_e1).cuda()
        
        # ranks, scores, score_of_expected, base, base_expected = ranker.forward(s, r, o, knowns, flag_eval=2)
        # ranks_e1, scores_e1, score_of_expected_e1, base_e1, base_expected_e1 = ranker.forward(s, r, o, knowns_e1, flag_eval=1)
        ranks, scores, score_of_expected = ranker.forward(s, r, o, knowns, entity_type_array, flag_eval=2)
        ranks_e1, scores_e1, score_of_expected_e1 = ranker.forward(s, r, o, knowns_e1, entity_type_array, flag_eval=1)
        # ranks, scores, score_of_expected = ranker.forward(s, r, o, knowns, flag_eval=2)
        # ranks_e1, scores_e1, score_of_expected_e1 = ranker.forward(s, r, o, knowns_e1, flag_eval=1)
        totals['e2']['mr'] += ranks.sum()
        totals['e2']['mrr'] += (1.0/ranks).sum()
        totals['e2']['hits10'] += ranks.le(11).float().sum()
        totals['e2']['hits1'] += ranks.eq(1).float().sum()
        #?,r,e2 
        totals['e1']['mr'] += ranks_e1.sum()
        totals['e1']['mrr'] += (1.0/ranks_e1).sum()
        totals['e1']['hits10'] += ranks_e1.le(11).float().sum()
        totals['e1']['hits1'] += ranks_e1.eq(1).float().sum()

        totals['m']['mr'] += (ranks.sum()+ranks_e1.sum())/2.0#((ranks+ranks_e1).sum()/2.0)
        totals['m']['mrr'] += ((1.0/ranks).sum()+(1.0/ranks_e1).sum())/2.0#((1.0/(ranks+ranks_e1)).sum()/2.0)
        totals['m']['hits10'] += (ranks_e1.le(11).float().sum() + ranks.le(11).float().sum())/2.0
        totals['m']['hits1'] += (ranks.eq(1).float().sum() + ranks_e1.eq(1).float().sum())/2.0
 
        extra = ""
        if verbose > 0:
            scores.scatter_(1, o.data, score_of_expected)
            top_scores, top_predictions = scores.topk(top_count, dim=-1)
            # base.scatter_(1, o.data, base_expected)
            # top_scores, top_predictions = base.topk(top_count, dim=-1)
            top_scores, top_predictions = scores.topk(top_count, dim=-1)
            top_predictions_type = torch.nn.functional.embedding(top_predictions, entity_type_matrix).squeeze(-1)
            expected_type = torch.nn.functional.embedding(o, entity_type_matrix).squeeze()
            correct = expected_type.eq(top_predictions_type[:, 0]).float()
            correct_count = correct.sum()
            totals['e2']["correct_type"] += correct_count[0]
            extra += " TP2 error %5.3f |" % (100*(1.0-totals['e2']["correct_type"]/end))
            for hook in hooks:
                hook(s.data, r.data, o.data, ranks, top_scores, top_predictions, expected_type, top_predictions_type)



            scores_e1.scatter_(1, s.data, score_of_expected_e1)
            top_scores_e1, top_predictions_e1 = scores_e1.topk(top_count, dim=-1)
            # base_e1.scatter_(1, s.data, base_expected_e1)
            # top_scores_e1, top_predictions_e1 = base_e1.topk(top_count, dim=-1)
            top_scores_e1, top_predictions_e1 = scores_e1.topk(top_count, dim=-1)
            top_predictions_type_e1 = torch.nn.functional.embedding(top_predictions_e1, entity_type_matrix).squeeze(-1)
            expected_type_e1 = torch.nn.functional.embedding(s, entity_type_matrix).squeeze()
            correct_e1 = expected_type_e1.eq(top_predictions_type_e1[:, 0]).float()
            correct_count_e1 = correct_e1.sum()
            totals['e1']["correct_type"] += correct_count_e1[0]
            extra += " TP1 error %5.3f |" % (100*(1.0-totals['e1']["correct_type"]/end))
            for hook in hooks:
                hook(s.data, r.data, o.data, ranks_e1, top_scores_e1, top_predictions_e1_e1, expected_type_e1, top_predictions_type_e1)
        
        #utils.print_progress_bar(end, facts.shape[0], "Evaluating on %s" % name, (("| mrr:%3.2f | mr:%10.5f | h10:%3.2f%"
        #                                                                          "% | h1:%3.2f%%| time %5.0f |") %
        #                         (100.0*totals['m']['mrr']/end, totals['m']['mr']/end, 100.0*totals['m']['hits10']/end,
        #                          100.0*totals['m']['hits1']/end, time.time()-start_time)) + extra, color="green")

        #utils.print_progress_bar(end, facts.shape[0], "Evaluating on %s" % name, (("|M| mrr:%3.2f | mr:%8.2f | h10:%3.2f%"
        #                                                                          "% | h1:%3.2f|e1| mrr:%3.2f | mr:%8.2f | h10:%3.2f%"
        #                                                                          "% | h1:%3.2f|e2| mrr:%3.2f | mr:%8.2f | h10:%3.2f%"
        #                                                                          "% | h1:%3.2f| time %5.0f |") %
        #                         (100.0*totals['m']['mrr']/end, totals['m']['mr']/end, 100.0*totals['m']['hits10']/end,
        #                          100.0*totals['m']['hits1']/end, 100.0*totals['e1']['mrr']/end, totals['e1']['mr']/end, 100.0*totals['e1']['hits10']/end,
        #                          100.0*totals['e1']['hits1']/end, 100.0*totals['e2']['mrr']/end, totals['e2']['mr']/end, 100.0*totals['e2']['hits10']/end,
        #                          100.0*totals['e2']['hits1']/end, time.time()-start_time)) + extra, color="green")

        utils.print_progress_bar(end, facts.shape[0], "Evaluating on %s" % name, (("|M| mrr:%3.2f | h10:%3.2f%"
                                                                                  "% | h1:%3.2f|e1| mrr:%3.2f| h10:%3.2f%"
                                                                                  "% | h1:%3.2f|e2| mrr:%3.2f | h10:%3.2f%"
                                                                                  "% | h1:%3.2f| time %5.0f |") %
                                 (100.0*totals['m']['mrr']/end, 100.0*totals['m']['hits10']/end,
                                  100.0*totals['m']['hits1']/end, 100.0*totals['e1']['mrr']/end, 100.0*totals['e1']['hits10']/end,
                                  100.0*totals['e1']['hits1']/end, 100.0*totals['e2']['mrr']/end, 100.0*totals['e2']['hits10']/end,
                                  100.0*totals['e2']['hits1']/end, time.time()-start_time)) + extra, color="green")
        # """
        # print()
        subject = numpy.vectorize(entity_name_map.get)(numpy.vectorize(reverse_entity_map.get)(s2))
        # pprint.pprint(subject)
        subject_type = numpy.vectorize(type_name_map.get)(expected_type_e1.cpu().numpy())
        # pprint.pprint(subject_type)
        # print()

        relation = numpy.vectorize(reverse_relation_map.get)(r2)
        # pprint.pprint(relation)
        # pprint.pprint(r[0])
        # pprint.pprint(r2)
        # pprint.pprint(reverse_relation_map[r.data.cpu().numpy()[0][0]])

        # print()
        object = numpy.vectorize(entity_name_map.get)(numpy.vectorize(reverse_entity_map.get)(o2))
        # pprint.pprint(object)
        object_type = numpy.vectorize(type_name_map.get)(expected_type.cpu().numpy())
        # pprint.pprint(object_type)
        # if i== 2:
        #     sys.exit()
        # """

        """
        #pprint.pprint(entity_name_map[reverse_entity_map[s2]])
        #pprint.pprint(reverse_entity_map[r2])
        #pprint.pprint(entity_name_map[reverse_entity_map[o2]])
        """
        # """
        # print()
        predictions_e1 = numpy.vectorize(entity_name_map.get)(numpy.vectorize(reverse_entity_map.get)(top_predictions_e1.cpu().numpy()))
        # pprint.pprint(predictions_e1)
        predictions_type_e1 = numpy.vectorize(type_name_map.get)(top_predictions_type_e1.cpu().numpy())
        # pprint.pprint(predictions_type_e1)
        
        # print()
        predictions = numpy.vectorize(entity_name_map.get)(numpy.vectorize(reverse_entity_map.get)(top_predictions.cpu().numpy()))
        # pprint.pprint(predictions)
        predictions_type = numpy.vectorize(type_name_map.get)(top_predictions_type.cpu().numpy()) 
        # pprint.pprint(predictions_type)
        # print()
        # """
        """
        print(expected_type_e1.data.cpu().numpy());

        print(type_name_map[int(expected_type_e1.data.cpu().numpy())]);

        print(expected_type.data.cpu().numpy());
        print(correct_e1.data.cpu().numpy()[0]);
        print(correct.data.cpu().numpy()[0]);

        if expected_type_e1.data[0] in subject_type_total:
                subject_type_total[expected_type_e1.data[0]] += 1
                subject_type_correct[expected_type_e1.data[0]] += correct_e1.data[0]  
        else:
                subject_type_total[expected_type_e1.data[0]] = 1
                subject_type_correct[expected_type_e1.data[0]] = correct_e1.data[0] 

        if expected_type.data[0] in object_type_total:
                object_type_total[expected_type.data[0]] += 1
                object_type_correct[expected_type.data[0]] += correct.data[0]  
        else:
                object_type_total[expected_type.data[0]] = 1
                object_type_correct[expected_type.data[0]] = correct.data[0] 
        """
        """
        #pprint.pprint(type_name_map[expected_type.cpu().numpy()])
        #pprint.pprint(type_name_map[top_predictions_type_e1.cpu().numpy()])
        #pprint.pprint(type_name_map[top_predictions_type.cpu().numpy()])
        """
        """
        pprint.pprint(ranks)
        pprint.pprint(scores)
        print(scores.shape)
        pprint.pprint(score_of_expected)
        pprint.pprint(top_scores)
        pprint.pprint(top_predictions)
        pprint.pprint(top_predictions_type)
        pprint.pprint(expected_type)
        pprint.pprint(correct)
        print()
        """
        """
        if i==2: 
                #for key in subject_type_total.keys():
                #    abs.append([key,subject_type_total[key],subject_type_correct[key],subject_type_correct[key]/subject_type_total[key]])

                with open("qwer.csv", "w") as f:
                    writer = csv.writer(f)
                    #writer.writerows(abs)
                    # writer.writerows([subject_type_total.keys(), subject_type_total.values()])
                    writer.writerows([relation_subject_type_total.keys(), relation_subject_type_total.values(), relation_subject_type_correct.keys(), relation_subject_type_correct.values(), relation_object_type_total.keys(), relation_object_type_total.values(), relation_object_type_correct.keys(), relation_object_type_correct.values()])

 
                print(all_rows)
                sys.exit()
        """
        ####\|/ this one works best
        # row = [subject[0], subject_type, relation[0], object[0], object_type, numpy.array2string(predictions_e1[0]), numpy.array2string(predictions_type_e1[0]), numpy.array2string(predictions[0]), numpy.array2string(predictions_type[0]), int(ranks_e1.cpu().numpy()[0]), int(ranks.cpu().numpy()[0]), int(correct_e1.cpu().numpy()[0]), int(correct.cpu().numpy()[0]) ]
        # row = [numpy.array2string(subject[0]), numpy.array2string(subject_type), numpy.array2string(relation[0]), numpy.array2string(object[0]), numpy.array2string(object_type), numpy.array2string(predictions_e1[0]), numpy.array2string(predictions_type_e1[0]), numpy.array2string(predictions[0]), numpy.array2string(predictions_type[0]), numpy.array2string(ranks_e1.cpu().numpy()[0]), numpy.array2string(ranks.cpu().numpy()[0]), numpy.array2string(correct_e1.cpu().numpy()[0]), numpy.array2string(correct.cpu().numpy()[0]) ]
        #row &= []
        #print(row)
        #print(numpy.asarray(subject,dtype=numpy.str_))
        #row = numpy.asarray(subject,dtype=numpy.str_)[0] + ',' + numpy.asarray(subject_type,dtype=numpy.str_) + ','
        #print(row)
        #row += Str(relation[0]) + ',' + Str(object[0]) + ',' +  Str(object_type) + ','
        #row += numpy.array2string(predictions_e1[0]) + ',' + numpy.array2string(predictions_type_e1[0]) + ',' + numpy.array2string(predictions[0]) + ',' + numpy.array2string(predictions_type_e1[0])

        #print(row)

        # all_rows.append(row)

        """
        print(expected_type_e1.data.cpu().numpy())

        print(type_name_map[int(expected_type_e1.data.cpu().numpy())])

        print(expected_type.data.cpu().numpy())
        print(correct_e1.data.cpu().numpy()[0])
        print(correct.data.cpu().numpy()[0])
        """
        
        #############WORKS
        """
        exp_type_e1 = type_name_map[int(expected_type_e1.data.cpu().numpy())]
        exp_type = type_name_map[int(expected_type.data.cpu().numpy())]

        cor_e1 = correct_e1.data.cpu().numpy()[0]
        cor = correct.data.cpu().numpy()[0]

        relation_name = reverse_relation_map[r.data.cpu().numpy()[0][0]]
        # print(relation_name)
        """
        
        """
        
        subject_name = entity_name_map[reverse_entity_map[s.data.cpu().numpy()[0][0]]]
        object_name = entity_name_map[reverse_entity_map[o.data.cpu().numpy()[0][0]]]
        # print(subject_name, object_name)

        subject_top_prediction_name = entity_name_map[reverse_entity_map[top_predictions_e1.data.cpu().numpy()[0][0]]]
        object_top_prediction_name = entity_name_map[reverse_entity_map[top_predictions.data.cpu().numpy()[0][0]]]
        # print(subject_top_prediction_name, object_top_prediction_name)

        # print(top_predictions_type_e1)
        # print(top_predictions_type_e1.data)
        # print(top_predictions_type_e1.data.cpu().numpy())
        subject_top_prediction_type = type_name_map[int(top_predictions_type_e1.data.cpu().numpy()[0][0])]
        object_top_prediction_type = type_name_map[int(top_predictions_type.data.cpu().numpy()[0][0])]
        # print(subject_top_prediction_type, object_top_prediction_type)


        gold_score, gold_base, gold_head, gold_tail = ranker.scoring_function.forward(s,r,o)
        sub_top_pred_score, sub_top_pred_base, sub_top_pred_head, sub_top_pred_tail = ranker.scoring_function.forward(top_predictions_e1.data[0][0],r,o)
        ob_top_pred_score, ob_top_pred_base, ob_top_pred_head, ob_top_pred_tail = ranker.scoring_function.forward(s,r,top_predictions.data[0][0])
        """
        """
        s1, b1, h1, t1 = ranker.scoring_function.forward(s, r, top_predictions.data[0][0])
        s2, b2, h2, t2 = ranker.scoring_function.forward(s, r, top_predictions.data[0][1])
        s3, b3, h3, t3 = ranker.scoring_function.forward(s, r, top_predictions.data[0][2])
        s4, b4, h4, t4 = ranker.scoring_function.forward(s, r, top_predictions.data[0][3])
        s5, b5, h5, t5 = ranker.scoring_function.forward(s, r, top_predictions.data[0][4])


        s1_e1, b1_e1, h1_e1, t1_e1 = ranker.scoring_function.forward(top_predictions_e1.data[0][0], r, o)
        s2_e1, b2_e1, h2_e1, t2_e1 = ranker.scoring_function.forward(top_predictions_e1.data[0][1], r, o)
        s3_e1, b3_e1, h3_e1, t3_e1 = ranker.scoring_function.forward(top_predictions_e1.data[0][2], r, o)
        s4_e1, b4_e1, h4_e1, t4_e1 = ranker.scoring_function.forward(top_predictions_e1.data[0][3], r, o)
        s5_e1, b5_e1, h5_e1, t5_e1 = ranker.scoring_function.forward(top_predictions_e1.data[0][4], r, o)

        row = [subject[0], subject_type, relation[0], object[0], object_type, numpy.array2string(predictions_e1[0]), numpy.array2string(predictions_type_e1[0]), numpy.array2string(predictions[0]), numpy.array2string(predictions_type[0]), int(ranks_e1.cpu().numpy()[0]), int(ranks.cpu().numpy()[0]), int(correct_e1.cpu().numpy()[0]), int(correct.cpu().numpy()[0]), gold_score.data.cpu().numpy()[0][0], gold_base.data.cpu().numpy()[0][0], gold_head.data.cpu().numpy()[0][0], gold_tail.data.cpu().numpy()[0][0], s1_e1.data.cpu().numpy()[0][0], b1_e1.data.cpu().numpy()[0][0], h1_e1.data.cpu().numpy()[0][0], t1_e1.data.cpu().numpy()[0][0], s1.data.cpu().numpy()[0][0], b1.data.cpu().numpy()[0][0], h1.data.cpu().numpy()[0][0], t1.data.cpu().numpy()[0][0], numpy.array2string(numpy.array([s1_e1.data.cpu().numpy()[0][0], s2_e1.data.cpu().numpy()[0][0], s3_e1.data.cpu().numpy()[0][0], s4_e1.data.cpu().numpy()[0][0], s5_e1.data.cpu().numpy()[0][0]])), numpy.array2string(numpy.array([b1_e1.data.cpu().numpy()[0][0], b2_e1.data.cpu().numpy()[0][0], b3_e1.data.cpu().numpy()[0][0], b4_e1.data.cpu().numpy()[0][0], b5_e1.data.cpu().numpy()[0][0]])), numpy.array2string(numpy.array([h1_e1.data.cpu().numpy()[0][0], h2_e1.data.cpu().numpy()[0][0], h3_e1.data.cpu().numpy()[0][0], h4_e1.data.cpu().numpy()[0][0], h5_e1.data.cpu().numpy()[0][0]])), numpy.array2string(numpy.array([t1_e1.data.cpu().numpy()[0][0], t2_e1.data.cpu().numpy()[0][0], t3_e1.data.cpu().numpy()[0][0], t4_e1.data.cpu().numpy()[0][0], t5_e1.data.cpu().numpy()[0][0]])), numpy.array2string(numpy.array([s1.data.cpu().numpy()[0][0], s2.data.cpu().numpy()[0][0], s3.data.cpu().numpy()[0][0], s4.data.cpu().numpy()[0][0], s5.data.cpu().numpy()[0][0]])), numpy.array2string(numpy.array([b1.data.cpu().numpy()[0][0], b2.data.cpu().numpy()[0][0], b3.data.cpu().numpy()[0][0], b4.data.cpu().numpy()[0][0], b5.data.cpu().numpy()[0][0]])), numpy.array2string(numpy.array([h1.data.cpu().numpy()[0][0], h2.data.cpu().numpy()[0][0], h3.data.cpu().numpy()[0][0], h4.data.cpu().numpy()[0][0], h5.data.cpu().numpy()[0][0]])), numpy.array2string(numpy.array([t1.data.cpu().numpy()[0][0], t2.data.cpu().numpy()[0][0], t3.data.cpu().numpy()[0][0], t4.data.cpu().numpy()[0][0], t5.data.cpu().numpy()[0][0]]))]

        all_rows.append(row)
        """
        
        # print(gold_score.data.cpu().numpy()[0][0], gold_base.data.cpu().numpy()[0][0], gold_head.data.cpu().numpy()[0][0], gold_tail.data.cpu().numpy()[0][0])
        # print(sub_top_pred_score.data.cpu().numpy()[0][0], sub_top_pred_base.data.cpu().numpy()[0][0], sub_top_pred_head.data.cpu().numpy()[0][0], sub_top_pred_tail.data.cpu().numpy()[0][0])
        # print(ob_top_pred_score.data.cpu().numpy()[0][0], ob_top_pred_base.data.cpu().numpy()[0][0], ob_top_pred_head.data.cpu().numpy()[0][0], ob_top_pred_tail.data.cpu().numpy()[0][0])

        # print(cor_e1,cor)
        # # print(ranks_e1.data.cpu().numpy(), ranks.data.cpu().numpy())
        # print(ranks_e1.data.cpu().numpy()[0], ranks.data.cpu().numpy()[0])
        ### Works
        # all_rows.append([relation_name, subject_name, exp_type_e1, subject_top_prediction_name, subject_top_prediction_type, int(ranks_e1.data.cpu().numpy()[0]), int(cor_e1), object_name, exp_type, object_top_prediction_name, object_top_prediction_type, int(ranks.data.cpu().numpy()[0]), int(cor), gold_score.data.cpu().numpy()[0][0], gold_base.data.cpu().numpy()[0][0], gold_head.data.cpu().numpy()[0][0], gold_tail.data.cpu().numpy()[0][0], sub_top_pred_score.data.cpu().numpy()[0][0], sub_top_pred_base.data.cpu().numpy()[0][0], sub_top_pred_head.data.cpu().numpy()[0][0], sub_top_pred_tail.data.cpu().numpy()[0][0], ob_top_pred_score.data.cpu().numpy()[0][0], ob_top_pred_base.data.cpu().numpy()[0][0], ob_top_pred_head.data.cpu().numpy()[0][0], ob_top_pred_tail.data.cpu().numpy()[0][0]])
        # all_rows.append([relation_name, subject_name, exp_type_e1, subject_top_prediction_name, subject_top_prediction_type, int(ranks_e1.data.cpu().numpy()[0]), int(cor_e1), object_name, exp_type, object_top_prediction_name, object_top_prediction_type, int(ranks.data.cpu().numpy()[0]), int(cor), gold_score.cpu().numpy()[0][0], gold_base.cpu().numpy()[0][0], gold_head.cpu().numpy()[0][0], gold_tail.cpu().numpy()[0][0], sub_top_pred_score.cpu().numpy()[0][0], sub_top_pred_base.cpu().numpy()[0][0], sub_top_pred_head.cpu().numpy()[0][0], sub_top_pred_tail.cpu().numpy()[0][0], ob_top_pred_score.cpu().numpy()[0][0], ob_top_pred_base.cpu().numpy()[0][0], ob_top_pred_head.cpu().numpy()[0][0], ob_top_pred_tail.cpu().numpy()[0][0]])
        ##########WORKS


        # if i==2:
        #     print(all_rows)
        #     sys.exit()
        # """

        ########################WORKS
        """
        if exp_type_e1 in subject_type_total:
                subject_type_total[exp_type_e1] += 1
                subject_type_correct[exp_type_e1] += cor_e1  
        else:
                subject_type_total[exp_type_e1] = 1
                subject_type_correct[exp_type_e1] = cor_e1 

        if exp_type in object_type_total:
                object_type_total[exp_type] += 1
                object_type_correct[exp_type] += cor  
        else:
                object_type_total[exp_type] = 1
                object_type_correct[exp_type] = cor



        if relation_name in relation_subject_type_total:
                relation_subject_type_total[relation_name] += 1
                relation_subject_type_correct[relation_name] += cor_e1  
        else:
                relation_subject_type_total[relation_name] = 1
                relation_subject_type_correct[relation_name] = cor_e1 

        if relation_name in relation_object_type_total:
                relation_object_type_total[relation_name] += 1
                relation_object_type_correct[relation_name] += cor  
        else:
                relation_object_type_total[relation_name] = 1
                relation_object_type_correct[relation_name] = cor
        """

    """
    with open("generated/entity_type_correct.csv", "w") as f:
        writer = csv.writer(f)
        #writer.writerows(abs)
        writer.writerows([subject_type_total.keys(), subject_type_total.values(), subject_type_correct.keys(), subject_type_correct.values(), object_type_total.keys(), object_type_total.values(), object_type_correct.keys(), object_type_correct.values()])

    with open("generated/relation_type_correct.csv", "w") as f:
        writer = csv.writer(f)
        #writer.writerows(abs)
        writer.writerows([relation_subject_type_total.keys(), relation_subject_type_total.values(), relation_subject_type_correct.keys(), relation_subject_type_correct.values(), relation_object_type_total.keys(), relation_object_type_total.values(), relation_object_type_correct.keys(), relation_object_type_correct.values()])

    """ 
    """
    relation_type_correctness = {}
    for x in relation_subject_type_correct.keys():
        relation_type_correctness[x] = {}
        relation_type_correctness[x]['m'] = ((relation_subject_type_correct[x]*100.0/relation_subject_type_total[x]) + (relation_object_type_correct[x]*100.0/relation_object_type_total[x]))/2.0
        relation_type_correctness[x]['s'] = relation_subject_type_correct[x]*100.0/relation_subject_type_total[x]
        relation_type_correctness[x]['o'] = relation_object_type_correct[x]*100.0/relation_object_type_total[x]

    entity_sub_type_correctness = {}
    for x in subject_type_total.keys():
        entity_sub_type_correctness[x] = subject_type_correct[x]*100.0/subject_type_total[x]

    entity_obj_type_correctness = {}
    for x in object_type_total.keys():
        entity_obj_type_correctness[x] = object_type_correct[x]*100.0/object_type_total[x]

    # print(((relation_subject_type_correct['/sports/sports_team/colors']*100.0/relation_subject_type_total['/sports/sports_team/colors']) + (relation_object_type_correct['/sports/sports_team/colors']*100.0/relation_object_type_total['/sports/sports_team/colors']))/2.0)
    # print(entity_sub_type_correctness['colors'] = subject_type_correct['colors']*100.0/subject_type_total['colors'])
    """


    """
    with open("generated/only_type_scores.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)
    """
    """
    as = []
    bs = []
    for key in subject_type_total:
        as.append([key,subject_type_total[key],subject_type_correct[key],subject_type_correct[key]/subject_type_total[key]])

    with open("qwer.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(as)
    """
    gc.collect()
    torch.cuda.empty_cache()
    for hook in hooks:
        hook.end()
    
    print("")

    totals['m'] = {x:totals['m'][x]/facts.shape[0] for x in totals['m']}
    totals['e1'] = {x:totals['e1'][x]/facts.shape[0] for x in totals['e1']}
    totals['e2'] = {x:totals['e2'][x]/facts.shape[0] for x in totals['e2']}

    # totals['e1']["correct_type"] *= facts.shape[0]
    # totals['e2']["correct_type"] *= facts.shape[0]

    # return totals#{x:totals['m'][x]/facts.shape[0] for x in totals['m']}
    return totals #, relation_type_correctness, entity_sub_type_correctness, entity_obj_type_correctness
