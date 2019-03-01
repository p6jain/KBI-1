'''
time CUDA_VISIBLE_DEVICES=1 python3 mukund_exp_get_pretrained_scores.py -d fb15k -m image_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}, "image_compatibility_coefficient":1}' -b 6500 -n 200 -v 1 -q 1 -f best_valid_model.pt 

x='-d fb15k -m image_model -a -b 6500 -n 200 -v 1 -q 1 -f best_valid_model.pt';L=x.split(" ")
L = L[:5] + ['{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}, "image_compatibility_coefficient":1}']+L[5:]
import sys
sys.argv += L

'''

import kb
import data_loader
import trainer
import torch
import losses
import models
import argparse
import os
import datetime
import json
import utils
import extra_utils
import pprint
import evaluate
#import mukund_evaluate_print as evaluate
#import exp_alpha_eval as evaluate
import csv
import numpy

has_cuda = torch.cuda.is_available()
if not has_cuda:
    utils.colored_print("yellow", "CUDA is not available, using cpu")


def main(dataset_root, model_name, model_arguments, batch_size, negative_sample_count, hooks,
         eval_batch_size, introduce_oov, verbose, saved_model_path):

    saved_model = torch.load(saved_model_path)

    entity_map, type_entity_range = saved_model['entity_map'], saved_model['type_entity_range']
    if 'reverse_entity_map' in saved_model:
        reverse_entity_map = saved_model['reverse_entity_map']
    else:
        reverse_entity_map = {}
        for k,v in entity_map.items():
            reverse_entity_map[v] = k

    ktrain = kb.kb(os.path.join(dataset_root, 'train.txt'), em=entity_map, type_entity_range=type_entity_range, rem=reverse_entity_map, add_unknowns=True)

    if introduce_oov:
        ktrain.entity_map["<OOV>"] = len(ktrain.entity_map)
    ktest = kb.kb(os.path.join(dataset_root, 'test.txt'), em=ktrain.entity_map, rm=ktrain.relation_map, rem=ktrain.reverse_entity_map, rrm=ktrain.reverse_relation_map,
                  add_unknowns=not introduce_oov)
    kvalid = kb.kb(os.path.join(dataset_root, 'valid.txt'), em=ktrain.entity_map, rm=ktrain.relation_map, rem=ktrain.reverse_entity_map, rrm=ktrain.reverse_relation_map,
                   add_unknowns=not introduce_oov)

    if(verbose > 0):
        print("train size", ktrain.facts.shape)
        print("test size", ktest.facts.shape)
        print("valid size", kvalid.facts.shape)
        utils.colored_print("yellow", "VERBOSE ANALYSIS only for FB15K")
        tpm = extra_utils.type_map_fine(dataset_root)
        enm = extra_utils.entity_name_map_fine(dataset_root)
        tnm = extra_utils.type_name_map_fine(dataset_root)
        ktrain.augment_type_information(tpm, enm, tnm)
        ktest.augment_type_information(tpm, enm, tnm)
        kvalid.augment_type_information(tpm, enm, tnm)
        hooks = extra_utils.load_hooks(hooks, ktrain)


    dltrain = data_loader.data_loader(ktrain, has_cuda)
    dlvalid = data_loader.data_loader(kvalid, has_cuda)
    dltest = data_loader.data_loader(ktest, has_cuda)


    model_arguments['entity_count'] = len(ktrain.entity_map)
    model_arguments['relation_count'] = len(ktrain.relation_map)

    if model_name == "image_model":
        model_arguments['image_embedding'] = numpy.load(dataset_root+"/image/image_embeddings_resnet152.dat")

    scoring_function = getattr(models, saved_model['model_name'])(**model_arguments)
    if has_cuda:
        scoring_function = scoring_function.cuda()

    if(not eval_batch_size):
        eval_batch_size = max(50, batch_size*2*negative_sample_count//len(ktrain.entity_map))

    scoring_function.load_state_dict(saved_model['model_weights'])

    print(saved_model['valid_score_m'])
    print(saved_model['valid_score_e1'])
    print(saved_model['valid_score_e2'])
    print(saved_model['test_score_m'])
    print(saved_model['test_score_e1'])
    print(saved_model['test_score_e2'])

    ranker = evaluate.ranker(scoring_function, kb.union([dltrain.kb, dlvalid.kb, dltest.kb]))
    
    indices = numpy.random.choice(ktrain.facts.shape[0], 10000, replace=False)
    facts = ktrain.facts[indices]
    ktrain.facts = facts

    scoring_function.eval()
    
    valid_score = evaluate.evaluate("valid", ranker, dlvalid.kb, eval_batch_size,
                                    verbose=verbose, hooks=hooks)
#    test_score = evaluate.evaluate("test ", ranker, dltest.kb, eval_batch_size,
                                   #verbose=verbose, hooks=hooks)
    valid_score["correct_type"]["e1"] = 100.0 - (100.0* valid_score["correct_type"]["e1"] / dlvalid.kb.facts.shape[0])
#    test_score["correct_type"]["e1"] = 100.0 - (100.0* test_score["correct_type"]["e1"] / dltest.kb.facts.shape[0])
    valid_score["correct_type"]["e2"] = 100.0 - (100.0* valid_score["correct_type"]["e2"] / dlvalid.kb.facts.shape[0])
#    test_score["correct_type"]["e2"] = 100.0 - (100.0* test_score["correct_type"]["e2"] / dltest.kb.facts.shape[0])

    print("Valid")
    pprint.pprint(valid_score)
#    print("Test")
#    pprint.pprint(test_score)
    """
    with open("generated/fb15k_rel_sigmoid.csv",'r') as f:
        reader = csv.reader(f)
        best_alpha_for_relation = dict(reader)

    train_mrr, train_e1_tc, train_e2_tc = evaluate.test_evaluate("train", ranker, ktrain, eval_batch_size, best_alpha_for_relation, verbose=verbose)
    print(train_mrr, train_e1_tc, train_e2_tc)

    # evaluate.colors("valid", ranker, dlvalid.kb, eval_batch_size, best_alpha_for_relation, verbose=verbose)
    valid_mrr, valid_e1_tc, valid_e2_tc = evaluate.test_evaluate("valid", ranker, dlvalid.kb, eval_batch_size, best_alpha_for_relation, verbose=verbose)
    #valid_mrr = evaluate.mrr_change_evaluate("valid", ranker, dlvalid.kb, eval_batch_size, best_alpha_for_relation, verbose=verbose)
    print(valid_mrr, valid_e1_tc, valid_e2_tc)

    test_mrr, test_e1_tc, test_e2_tc = evaluate.test_evaluate("test", ranker, dltest.kb, eval_batch_size, best_alpha_for_relation, verbose=verbose)
    print(test_mrr, test_e1_tc, test_e2_tc)

    print("valid", valid_mrr, valid_e1_tc, valid_e2_tc)
    print("test", test_mrr, test_e1_tc, test_e2_tc)
    
    """
    """
    bafrn, rel_mrr, mrr = evaluate.calculate_alpha("valid", ranker, dlvalid.kb, eval_batch_size,verbose=verbose)
    print(bafrn)
    print(mrr)

    with open("generated/fb15k_rel_alpha2.csv",'w') as f:
        writer = csv.writer(f)
        for k,v in bafrn.items():
            writer.writerow([k,v])
 
    #with open("generated/fb15k_l2_rel_mrr_alpha.csv",'w') as f:
    #    writer = csv.writer(f)
    #    for k,v in rel_mrr.items():
    #        writer.writerow([k,v])
    """
    """ 
    bbfrn, rel_mrr, mrr = evaluate.calculate_beta("valid", ranker, dlvalid.kb, eval_batch_size,verbose=verbose)
    print(bbfrn)
    print(mrr)

    with open("generated/fb15k_rel_beta.csv",'w') as f:
        writer = csv.writer(f)
        for k,v in bbfrn.items():
            writer.writerow([k,v])
    """
    """ 
    bbfrn, rel_mrr, mrr = evaluate.calculate_sigmoid("valid", ranker, dlvalid.kb, eval_batch_size,verbose=verbose)
    print(bbfrn)
    print(mrr)

    with open("generated/fb15k_rel_sigmoid.csv",'w') as f:
        writer = csv.writer(f)
        for k,v in bbfrn.items():
            writer.writerow([k,v])
    """



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help="Name of the dataset as in data folder", required=True)
    parser.add_argument('-m', '--model', help="model name as in models.py", required=True)
    parser.add_argument('-a', '--model_arguments', help="model arguments as in __init__ of "
                                                        "model (Excluding entity and relation count) "
                                                        "This is a json string", required=True)
    parser.add_argument('-b', '--batch_size', required=False, type=int, default=2000)
    parser.add_argument('-y', '--eval_batch_size', required=False, type=int, default=0)
    parser.add_argument('-n', '--negative_sample_count', required=False, type=int, default=200)
    parser.add_argument('-v', '--oov_entity', required=False)
    parser.add_argument('-q', '--verbose', required=False, default=0, type=int)
    parser.add_argument('-k', '--hooks', required=False, default="[]")
    parser.add_argument('-f', '--saved_model_path', required=False)
    parser.add_argument('--data_repository_root', required=False, default='data')
    
    arguments = parser.parse_args()
    
    arguments.model_arguments = json.loads(arguments.model_arguments)
    arguments.hooks = json.loads(arguments.hooks)
    print(arguments)
    
    dataset_root = os.path.join(arguments.data_repository_root, arguments.dataset)
    
    main(dataset_root, arguments.model, arguments.model_arguments, arguments.batch_size, arguments.negative_sample_count, arguments.hooks, arguments.eval_batch_size, arguments.oov_entity, arguments.verbose, arguments.saved_model_path)
