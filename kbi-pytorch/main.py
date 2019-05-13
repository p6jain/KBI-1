'''
time CUDA_VISIBLE_DEVICES=0 python3 main.py -d fb15k -m image_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}, "image_compatibility_coefficient":1}' -l softmax_loss -r 0.5 -g 0.30 -b 4500 -x 2000 -n 200 -v 1 -y 25 -e 500 -q 1
time CUDA_VISIBLE_DEVICES=0 python3 main.py -d fb15k -m image_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}, "image_compatibility_coefficient":1, "additional_params":{"flag_use_image":1}}' -l softmax_loss -r 0.5 -g 0.30 -b 4500 -x 2000 -n 200 -v 1 -y 25 -e 500 -q 1


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
import numpy
import re
import sys
import pickle
import torch.optim.lr_scheduler as lr_scheduler 

has_cuda = torch.cuda.is_available()
if not has_cuda:
    utils.colored_print("yellow", "CUDA is not available, using cpu")


def main(dataset_root, save_dir, model_name, model_arguments, loss_function, learning_rate, batch_size,
         regularization_coefficient, gradient_clip, optimizer_name, max_epochs, negative_sample_count, hooks,
         eval_every_x_mini_batches, eval_batch_size, resume_from_save, introduce_oov, verbose):

    print("Prachi Debug!!", "Using LR scheduler")

    flag_add_reverse = 1 if re.search("_icml", model_name) else 0
    model_name = model_name.split("_icml")[0]

    flag_image = 0
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

    tpm = None
    if verbose>0:
        utils.colored_print("yellow", "VERBOSE ANALYSIS only for FB15K")
        tpm = extra_utils.type_map_fine(dataset_root)

    if resume_from_save:
        saved_model = torch.load(resume_from_save)
        entity_map = saved_model['entity_map']
    else:
        entity_map = {}
    
    if flag_image:
        #base_model_arguments
        print("Prachi Debug 1:", model_arguments)
        additional_params = model_arguments["additional_params"]
        model_arguments.pop("additional_params")
        print("Prachi Debug 1:", model_arguments)
        #"flag_use_image", "flag_facts_with_image","flag_reg_penalty_only_images","flag_reg_penalty_ent_prob","flag_reg_penalty_image_compat"
        ktrain = kb.kb(os.path.join(dataset_root, 'train.txt'), em=entity_map, im_em=im_entity_map, im_rem=im_reverse_entity_map, additional_params=additional_params)#{'flag_use_image':1})
        if introduce_oov and not "<OOV>" in ktrain.entity_map.keys():
            ktrain.entity_map["<OOV>"] = len(ktrain.entity_map)
            ktrain.mid_imid_map[ktrain.entity_map["<OOV>"]] = im_entity_map["<OOV>"]
            ktrain.nonoov_entity_count = ktrain.entity_map["<OOV>"]+1          
         
        ktest = kb.kb(os.path.join(dataset_root, 'test.txt'), em=ktrain.entity_map, rm=ktrain.relation_map, 
                   rem=ktrain.reverse_entity_map, rrm=ktrain.reverse_relation_map,
                   im_em=ktrain.im_entity_map, im_rem=ktrain.im_reverse_entity_map, mid_imid = ktrain.mid_imid_map,
                   add_unknowns=1, additional_params=additional_params, nonoov_entity_count=ktrain.entity_map["<OOV>"]+1)#{'flag_use_image':1})

        kvalid = kb.kb(os.path.join(dataset_root, 'valid.txt'), em=ktrain.entity_map, rm=ktrain.relation_map, 
                   rem=ktrain.reverse_entity_map, rrm=ktrain.reverse_relation_map,
                   im_em=ktrain.im_entity_map, im_rem=ktrain.im_reverse_entity_map, mid_imid = ktrain.mid_imid_map,
                   add_unknowns=1, additional_params=additional_params, nonoov_entity_count=ktrain.entity_map["<OOV>"]+1)#{'flag_use_image':1})

        '''
        ktest = kb.kb(os.path.join(dataset_root, 'test.txt'), em=ktrain.entity_map, rm=ktrain.relation_map, 
                   rem=ktrain.reverse_entity_map, rrm=ktrain.reverse_relation_map,
                   im_em=ktrain.im_entity_map, im_rem=ktrain.im_reverse_entity_map, mid_imid = ktrain.mid_imid_map,
                   add_unknowns=not introduce_oov, additional_params=additional_params)#{'flag_use_image':1})

        kvalid = kb.kb(os.path.join(dataset_root, 'valid.txt'), em=ktrain.entity_map, rm=ktrain.relation_map, 
                   rem=ktrain.reverse_entity_map, rrm=ktrain.reverse_relation_map,
                   im_em=ktrain.im_entity_map, im_rem=ktrain.im_reverse_entity_map, mid_imid = ktrain.mid_imid_map,
                   add_unknowns=not introduce_oov, additional_params=additional_params)#{'flag_use_image':1})
        '''
    else:
        ktrain = kb.kb(os.path.join(dataset_root, 'train.txt'), em=entity_map)
        if introduce_oov: 
            if not "<OOV>" in ktrain.entity_map.keys():
                ktrain.entity_map["<OOV>"] = len(ktrain.entity_map)
            ktrain.nonoov_entity_count = ktrain.entity_map["<OOV>"]+1
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
        print("num entities", len(ktrain.entity_map))
        print("num non-oov ent", ktrain.nonoov_entity_count)
        print("num relations", len(ktrain.relation_map))
        enm = extra_utils.entity_name_map_fine(dataset_root)
        tnm = extra_utils.type_name_map_fine(dataset_root)
        ktrain.augment_type_information(tpm,enm,tnm)
        ktest.augment_type_information(tpm,enm,tnm)
        kvalid.augment_type_information(tpm,enm,tnm)
        hooks = extra_utils.load_hooks(hooks, ktrain)

    if flag_add_reverse:#data for reg
        ktrain.augment_prob_information()
        kvalid.augment_prob_information(e_p=ktrain.e_prob, r_p = ktrain.r_prob)#not req for reg
        ktest.augment_prob_information(e_p=ktrain.e_prob, r_p = ktrain.r_prob)#not req for reg

    #if model_name == "image_model":
        #eim = extra_utils.fb15k_entity_image_map()
        #ktrain.augment_image_information(eim)
        #print("Prachi Debug::","3")
        #ktest.augment_image_information(eim)
        #print("Prachi Debug::","4")
        #kvalid.augment_image_information(eim)
        #print("Prachi Debug::","5")

    if introduce_oov:
        model_arguments['entity_count'] = ktrain.entity_map["<OOV>"] + 1#len(ktrain.entity_map)
    else:
        model_arguments['entity_count'] = len(ktrain.entity_map)
    if flag_add_reverse:
        model_arguments['relation_count'] = len(ktrain.relation_map)*2
        model_arguments['flag_add_reverse'] = flag_add_reverse
    else:
        model_arguments['relation_count'] = len(ktrain.relation_map)

    dltrain = data_loader.data_loader(ktrain, has_cuda, flag_add_reverse=flag_add_reverse, loss=loss_function, num_entity = model_arguments['entity_count'])
    dlvalid = data_loader.data_loader(kvalid, has_cuda, flag_add_reverse=flag_add_reverse, loss=loss_function, num_entity = model_arguments['entity_count'])
    dltest = data_loader.data_loader(ktest, has_cuda, flag_add_reverse=flag_add_reverse, loss=loss_function, num_entity = model_arguments['entity_count'])



    '''
    if model_name == "image_model" or model_name == "only_image_model" or model_name == "typed_image_model" or model_name =="typed_image_model_reg":
        #model_arguments['image_embedding'] = numpy.load(dataset_root+"/image/image_embeddings_resnet152.dat") #dump dictionary
        ''''''
        To do:
        load a dic now mid - image embed 
        use ktrain.im_entity_map for building a matrix to be used
        use may dump the im_entity_map and the matrix for reuse later!! 
        ''''''
        self.kvalid.im_entity_map["<OOV>"] = len(self.kvalid.im_entity_map)
        print("check all same", len(self.ktrain.im_entity_map),len(self.ktest.im_entity_map),len(self.kvalid.im_entity_map))
        with open(dataset_root+"/image/image_embeddings_resnet152_dict-mid-imageE.pkl","rb") as in_pkl:
            mid_ie_map = pickle.load(in_pkl)

        size_details = tuple([len(mid_ie_map)]+list(mid_ie_map[list(mid_ie_map.keys())[0]].shape[1:]))
        entity_id_image_matrix = numpy.zeros(size_details)
        oov_image=numpy.random.rand(1, 3, 224, 224);oov_count=0
        for x in self.kvalid.im_entity_map:
            if x in entity_mid_image_map.keys():
                entity_id_image_matrix[entity_map[x]] = entity_mid_image_map[x]
            else:
                entity_id_image_matrix[entity_map[x]] = oov_image
                oov_count+=1
 

         model_arguments['image_embedding'] = 
     '''


    if 0:#model_name == "typed_model_v2":
        best_beta = extra_utils.get_betas(dataset_root, ktrain.relation_map)
        print("Prachi Debug:","best beta",best_beta)
        model_arguments['best_beta'] = best_beta

    scoring_function = getattr(models, model_name)(**model_arguments)
    if has_cuda:
        scoring_function = scoring_function.cuda()
    loss = getattr(losses, loss_function)()
    optim = getattr(torch.optim, optimizer_name)(scoring_function.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optim, 'max', patience = 2, verbose=True)#mrr tracking 

    if(not eval_batch_size):
        eval_batch_size = max(50, batch_size*2*negative_sample_count//len(ktrain.entity_map))

    if model_name == "image_model" or model_name == "typed_image_model" or model_name == "typed_image_model_reg":
        tr = trainer.Trainer(scoring_function, scoring_function.regularizer, loss, optim, dltrain, dlvalid, dltest,
                         batch_size=batch_size, eval_batch=eval_batch_size, negative_count=negative_sample_count,
                         save_dir=save_dir, gradient_clip=gradient_clip, hooks=hooks,
                         regularization_coefficient=regularization_coefficient, verbose=verbose, model_name=model_name,
                         image_compatibility = scoring_function.image_compatibility, image_compatibility_coefficient = scoring_function.image_compatibility_coefficient, scheduler=scheduler)#0.01)
    elif flag_add_reverse:
        print("Prachi Info::", "using icml reg", "regularizer_icml")#"regularizer_icml")#regularizer_icml_orig
        tr = trainer.Trainer(scoring_function, scoring_function.regularizer_icml, loss, optim, dltrain, dlvalid, dltest,
                         batch_size=batch_size, eval_batch=eval_batch_size, negative_count=negative_sample_count,
                         save_dir=save_dir, gradient_clip=gradient_clip, hooks=hooks,
                         regularization_coefficient=regularization_coefficient, verbose=verbose, scheduler=scheduler)
    else:
        tr = trainer.Trainer(scoring_function, scoring_function.regularizer_icml, loss, optim, dltrain, dlvalid, dltest,
                         batch_size=batch_size, eval_batch=eval_batch_size, negative_count=negative_sample_count,
                         save_dir=save_dir, gradient_clip=gradient_clip, hooks=hooks,
                         regularization_coefficient=regularization_coefficient, verbose=verbose, scheduler=scheduler)


    if resume_from_save:
        mb_start = tr.load_state(resume_from_save)
        #m2 = "logs/complex {'embedding_dim': 1000, 'clamp_v': 1.0} crossentropy_loss run on yago starting from 2019-05-09 12:02:02.715809/best_valid_model.pt"
        #mb_start = tr.load_state(m2)
    else:
        mb_start = 0
    max_mini_batch_count = int(max_epochs*ktrain.facts.shape[0]/batch_size)
    print("max_mini_batch_count: %d, eval_batch_size %d" % (max_mini_batch_count, eval_batch_size))
    tr.start(max_mini_batch_count, [eval_every_x_mini_batches//20, 20], mb_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help="Name of the dataset as in data folder", required=True)
    parser.add_argument('-m', '--model', help="model name as in models.py", required=True)
    parser.add_argument('-a', '--model_arguments', help="model arguments as in __init__ of "
                                                        "model (Excluding entity and relation count) "
                                                        "This is a json string", required=True)
    parser.add_argument('-o', '--optimizer', required=False, default='Adagrad')
    parser.add_argument('-l', '--loss', help="loss function name as in losses.py", required=True)
    parser.add_argument('-r', '--learning_rate', required=True, type=float)
    parser.add_argument('-g', '--regularization_coefficient', required=True, type=float)
    parser.add_argument('-c', '--gradient_clip', required=False, type=float)
    parser.add_argument('-e', '--max_epochs', required=False, type=int, default=1000)
    parser.add_argument('-b', '--batch_size', required=False, type=int, default=2000)
    parser.add_argument('-x', '--eval_every_x_mini_batches', required=False, type=int, default=2000)
    parser.add_argument('-y', '--eval_batch_size', required=False, type=int, default=0)
    parser.add_argument('-n', '--negative_sample_count', required=False, type=int, default=200)
    parser.add_argument('-s', '--save_dir', required=False)
    parser.add_argument('-u', '--resume_from_save', required=False)
    parser.add_argument('-v', '--oov_entity', required=False)
    parser.add_argument('-q', '--verbose', required=False, default=0, type=int)
    parser.add_argument('-z', '--debug', required=False, default=0, type=int)
    parser.add_argument('-k', '--hooks', required=False, default="[]")
    parser.add_argument('--data_repository_root', required=False, default='data')
    parser.add_argument('-msg', '--message', required=False, default="")
    arguments = parser.parse_args()
    time = str(datetime.datetime.now().isoformat(' ', 'seconds'))
    if arguments.save_dir is None:
        tmp = json.loads(arguments.model_arguments);
        if "additional_params" in tmp.keys():
            tmp.pop("additional_params")
        tmp  = str(tmp)

        arguments.save_dir = os.path.join("logs", "%s %s %s run on %s starting from %s" % (arguments.model,
                                                                                        tmp,#arguments.model_arguments,
                                                                                        arguments.loss,
                                                                                        arguments.dataset,
                                                                                        str(datetime.datetime.now())))
        '''
        tmp = json.loads(arguments.model_arguments);
        if "additional_params" in tmp.keys():
            tmp.pop("additional_params")
        tmp  = str(tmp)
        arguments.save_dir = os.path.join("logs", "%s %s run on %s starting from %s" % (arguments.model,
                                                                                        tmp,#arguments.model_arguments,
                                                                                        #arguments.loss,
                                                                                        arguments.dataset,
                                                                                        time))
        
        '''
    arguments.model_arguments = json.loads(arguments.model_arguments)
    arguments.hooks = json.loads(arguments.hooks)
    if not arguments.debug:
        if not os.path.isdir(arguments.save_dir):
            print("Making directory (s) %s" % arguments.save_dir)
            os.makedirs(arguments.save_dir)
        else:
            utils.colored_print("yellow", "directory %s already exists" % arguments.save_dir)
        utils.duplicate_stdout(os.path.join(arguments.save_dir, "log.txt"))
    print(arguments)
    print("User Message:: ", arguments.message)
    print("Command:: ", (" ").join(sys.argv))
    dataset_root = os.path.join(arguments.data_repository_root, arguments.dataset)
    main(dataset_root, arguments.save_dir, arguments.model, arguments.model_arguments, arguments.loss,
         arguments.learning_rate, arguments.batch_size,arguments.regularization_coefficient, arguments.gradient_clip,
         arguments.optimizer, arguments.max_epochs, arguments.negative_sample_count, arguments.hooks,
         arguments.eval_every_x_mini_batches, arguments.eval_batch_size, arguments.resume_from_save,
         arguments.oov_entity, arguments.verbose)
