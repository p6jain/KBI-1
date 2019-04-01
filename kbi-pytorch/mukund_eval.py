'''
time CUDA_VISIBLE_DEVICES=1 python3 mukund_exp_get_pretrained_scores.py -d fb15k -m image_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}, "image_compatibility_coefficient":1}' -b 6500 -n 200 -v 1 -q 1 -f best_valid_model.pt 

x='-d fb15k -m image_model -a -b 6500 -n 200 -v 1 -q 1 -f best_valid_model.pt';L=x.split(" ")
L = L[:5] + ['{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}, "image_compatibility_coefficient":1}']+L[5:]
import sys
sys.argv += L

time CUDA_VISIBLE_DEVICES=4 python3 mukund_eval.py -d fb15k -m typed_image_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}, "image_compatibility_coefficient":0}' -b 6500 -n 200 -v 1 -q 1 -f "logs/typed_image_model {'embedding_dim': 19, 'base_model_name': 'complex', 'base_model_arguments': {'embedding_dim': 180}, 'image_compatibility_coefficient': 0} softmax_loss run on fb15k starting from 2019-03-27 07:31:04.849531/best_valid_model.pt" -y 50


'''
#import mukund_kb as kb
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
#import mukund_freq_evaluate as evaluate
import csv
import numpy
from collections import Counter
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import pickle

has_cuda = torch.cuda.is_available()
if not has_cuda:
    utils.colored_print("yellow", "CUDA is not available, using cpu")


def main(dataset_root, model_name, model_arguments, batch_size, negative_sample_count, hooks,
         eval_batch_size, introduce_oov, verbose, saved_model_path):
    #Load Model
    saved_model = torch.load(saved_model_path)

    entity_map = saved_model['entity_map']
    if 'reverse_entity_map' in saved_model:
        reverse_entity_map = saved_model['reverse_entity_map']
    else:
        reverse_entity_map = {}
        for k,v in entity_map.items():
            reverse_entity_map[v] = k
    #relation_map = saved_model['relation_map']

    flag_image = 0

    #Load images
    if model_name == "image_model" or model_name == "only_image_model" or model_name == "typed_image_model" or model_name == "typed_image_model_reg":
        flag_image = 1
        with open(dataset_root+"/image/image_embeddings_resnet152_new_mid_iid.pkl","rb") as f:
            im_entity_map = pickle.load(f)#mid to iid 
        im_reverse_entity_map = {im_entity_map[ele]: ele for ele in im_entity_map}   #iid to mid

        image_embedding = numpy.load(dataset_root+"/image/image_embeddings_resnet152_new.dat") ###add
        oov_random_embedding = numpy.random.rand(1,image_embedding.shape[-1])
        image_embedding = numpy.vstack((image_embedding,oov_random_embedding))
        model_arguments['image_embedding'] = image_embedding
        oov_id = len(im_entity_map)
        im_entity_map["<OOV>"] = oov_id; im_reverse_entity_map[oov_id] = "<OOV>"
        print("OOV ID here", oov_id)
    
    #Load aux data
    tpm = None
    if verbose>0:
        utils.colored_print("yellow", "VERBOSE ANALYSIS only for FB15K")                                                                            
        tpm = extra_utils.type_map_fine(dataset_root) 

    #Prepare data
    if flag_image:
        #load more data
        mid_imid_map = saved_model['mid_imid_map']
        additional_params = saved_model["additional_params"]
        #"flag_use_image", "flag_facts_with_image","flag_reg_penalty_only_images","flag_reg_penalty_ent_prob","flag_reg_penalty_image_compat"
        ktrain = kb.kb(os.path.join(dataset_root, 'train.txt'), em=entity_map, im_em=im_entity_map, im_rem=im_reverse_entity_map, mid_imid = mid_imid_map, additional_params=additional_params)
        if introduce_oov and not "<OOV>" in ktrain.entity_map.keys():
            ktrain.entity_map["<OOV>"] = len(ktrain.entity_map)
            ktrain.mid_imid_map[ktrain.entity_map["<OOV>"]] = im_entity_map["<OOV>"]
            ktrain.nonoov_entity_count = ktrain.entity_map["<OOV>"]+1
        else:
            ktrain.nonoov_entity_count = saved_model['nonoov_entity_count']
        ktest = kb.kb(os.path.join(dataset_root, 'test.txt'), em=ktrain.entity_map, rm=ktrain.relation_map,
                   rem=ktrain.reverse_entity_map, rrm=ktrain.reverse_relation_map,
                   im_em=ktrain.im_entity_map, im_rem=ktrain.im_reverse_entity_map, mid_imid = ktrain.mid_imid_map,
                   add_unknowns=1, additional_params=additional_params, nonoov_entity_count=ktrain.entity_map["<OOV>"]+1)#{'flag_use_image':1})

        kvalid = kb.kb(os.path.join(dataset_root, 'valid.txt'), em=ktrain.entity_map, rm=ktrain.relation_map,
                   rem=ktrain.reverse_entity_map, rrm=ktrain.reverse_relation_map,
                   im_em=ktrain.im_entity_map, im_rem=ktrain.im_reverse_entity_map, mid_imid = ktrain.mid_imid_map,
                   add_unknowns=1, additional_params=additional_params, nonoov_entity_count=ktrain.entity_map["<OOV>"]+1)#{'flag_use_image':1})
    else:
        ktrain = kb.kb(os.path.join(dataset_root, 'train.txt'), em=entity_map)
        if introduce_oov and not "<OOV>" in ktrain.entity_map.keys():
            ktrain.entity_map["<OOV>"] = len(ktrain.entity_map)
        ktrain.nonoov_entity_count = ktrain.entity_map["<OOV>"]+1
        print("Prachi Debug", len(ktrain.relation_map), len(ktrain.entity_map)) 
        ktest = kb.kb(os.path.join(dataset_root, 'test.txt'), em=ktrain.entity_map, rm=ktrain.relation_map,
                   rem=ktrain.reverse_entity_map, rrm=ktrain.reverse_relation_map,
                   add_unknowns=not introduce_oov, nonoov_entity_count=ktrain.entity_map["<OOV>"]+1)
        kvalid = kb.kb(os.path.join(dataset_root, 'valid.txt'), em=ktrain.entity_map, rm=ktrain.relation_map,
                   rem=ktrain.reverse_entity_map, rrm=ktrain.reverse_relation_map,
                   add_unknowns=not introduce_oov, nonoov_entity_count=ktrain.entity_map["<OOV>"]+1)

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


    if introduce_oov:
        if flag_image:
            model_arguments['entity_count'] = ktrain.entity_map["<OOV>"] + 1
        else:
            model_arguments['entity_count'] = len(ktrain.entity_map)
    else:
        model_arguments['entity_count'] = len(ktrain.entity_map)
    model_arguments['relation_count'] = len(ktrain.relation_map)

    scoring_function = getattr(models, saved_model['model_name'])(**model_arguments)
    if has_cuda:
        scoring_function = scoring_function.cuda()

    if(not eval_batch_size):
        eval_batch_size = max(50, batch_size*2*negative_sample_count//len(ktrain.entity_map))

    '''
    print(saved_model)
    for ele in saved_model.keys():
        if type(saved_model[ele])==dict:
            print(ele,len(saved_model[ele]))
            continue
        try:
            print(ele,saved_model[ele].shape)
        except:
            print(ele,saved_model[ele])
    '''
    for ele in saved_model["model_weights"].keys():
        print(ele, saved_model["model_weights"][ele].shape)
    scoring_function.load_state_dict(saved_model['model_weights'])

    print("valid_score_m",saved_model['valid_score_m'])
    print("valid_score_e1", saved_model['valid_score_e1'])
    print("valid_score_e2", saved_model['valid_score_e2'])
    print("test_score_m", saved_model['test_score_m'])
    print("test_score_e1", saved_model['test_score_e1'])
    print("test_score_e2", saved_model['test_score_e2'])

    ranker = evaluate.ranker(scoring_function, kb.union([dltrain.kb, dlvalid.kb, dltest.kb]))
    valid_score = evaluate.evaluate("valid", ranker, dlvalid.kb, eval_batch_size,
                                    verbose=verbose, hooks=hooks, save=1)
    test_score = evaluate.evaluate("test ", ranker, dltest.kb, eval_batch_size,
                                   verbose=verbose, hooks=hooks, save=0)
    valid_score["correct_type"]["e1"] = 100.0 - (100.0* valid_score["correct_type"]["e1"] / dlvalid.kb.facts.shape[0])
    test_score["correct_type"]["e1"] = 100.0 - (100.0* test_score["correct_type"]["e1"] / dltest.kb.facts.shape[0])
    valid_score["correct_type"]["e2"] = 100.0 - (100.0* valid_score["correct_type"]["e2"] / dlvalid.kb.facts.shape[0])
    test_score["correct_type"]["e2"] = 100.0 - (100.0* test_score["correct_type"]["e2"] / dltest.kb.facts.shape[0])

    print("Valid")
    pprint.pprint(valid_score)
    print("Test")
    pprint.pprint(test_score)
    """
    sub_freq = dict(Counter(dltrain.kb.facts[:,0]))
    print(sub_freq[0])
    print("num_ents", len(dltrain.kb.entity_map.keys()))

    freqs = torch.zeros(len(dltrain.kb.entity_map.keys()), dtype=torch.float)
    print("freqs shape", freqs.shape)
    print("freqs before", freqs)
    for k,v in sub_freq.items():
        freqs[k] = v

    print("freqs after", freqs)

    mrr, e1tc, e2tc, bucket_results = evaluate.freq_test_evaluate("valid", ranker, dlvalid.kb, eval_batch_size, freqs, verbose=1, top_count=1)

    print(mrr, e1tc, e2tc, bucket_results)
    with open("bucket.tsv",'w') as outfile:
        csv_writer = csv.writer(outfile, delimiter='\t') 
        for k,v in bucket_results.items():
            csv_writer.writerow([k, v[0], v[1], v[2], v[3]])

    """
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
