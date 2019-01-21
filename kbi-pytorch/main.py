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



has_cuda = torch.cuda.is_available()
if not has_cuda:
    utils.colored_print("yellow", "CUDA is not available, using cpu")


def main(dataset_root, save_dir, model_name, model_arguments, loss_function, learning_rate, batch_size,
         regularization_coefficient, gradient_clip, optimizer_name, max_epochs, negative_sample_count, hooks,
         eval_every_x_mini_batches, eval_batch_size, resume_from_save, introduce_oov, verbose):
    #print("Prachi::", "with distance margin 0.0, distance dot like l2 ent type")#, ent reg")

    tpm = None
    if verbose>0:
        utils.colored_print("yellow", "VERBOSE ANALYSIS only for FB15K")
        tpm = extra_utils.fb15k_type_map_fine()

    if resume_from_save:
        saved_model = torch.load(resume_from_save)
        entity_map, type_entity_range = saved_model['entity_map'], saved_model['type_entity_range']
        if 'reverse_entity_map' in saved_model:
            reverse_entity_map = saved_model['reverse_entity_map']
        else:
            reverse_entity_map = {}
            for k,v in entity_map.items():
                reverse_entity_map[v] = k

    else:
        entity_map, reverse_entity_map, type_entity_range = extra_utils.get_entity_relation_id_neg_sensitive(tpm)

    ktrain = kb.kb(os.path.join(dataset_root, 'train.txt'), em=entity_map, type_entity_range=type_entity_range, rem=reverse_entity_map)

    if introduce_oov:
        ktrain.entity_map["<OOV>"] = len(ktrain.entity_map)
    ktest = kb.kb(os.path.join(dataset_root, 'test.txt'), em=ktrain.entity_map, rm=ktrain.relation_map, rem=ktrain.reverse_entity_map, rrm=ktrain.reverse_relation_map,
                  add_unknowns=not introduce_oov)
    kvalid = kb.kb(os.path.join(dataset_root, 'valid.txt'), em=ktrain.entity_map, rm=ktrain.relation_map, rem=ktrain.reverse_entity_map, rrm=ktrain.reverse_relation_map,
                   add_unknowns=not introduce_oov)

    if(verbose > 0):
        enm = extra_utils.fb15k_entity_name_map_fine()
        tnm = extra_utils.fb15k_type_name_map_fine()
        ktrain.augment_type_information(tpm,enm,tnm)
        ktest.augment_type_information(tpm,enm,tnm)
        kvalid.augment_type_information(tpm,enm,tnm)
        hooks = extra_utils.load_hooks(hooks, ktrain)

    if model_name == "image_model":
        #print("Prachi Debug::","1")
        eim = extra_utils.fb15k_entity_image_map()
        #print("Prachi Debug::","2")
        ktrain.augment_image_information(eim)
        #print("Prachi Debug::","3")
        #ktest.augment_image_information(eim)
        #print("Prachi Debug::","4")
        #kvalid.augment_image_information(eim)
        #print("Prachi Debug::","5")

    dltrain = data_loader.data_loader(ktrain, has_cuda)
    dlvalid = data_loader.data_loader(kvalid, has_cuda)
    dltest = data_loader.data_loader(ktest, has_cuda)

    model_arguments['entity_count'] = len(ktrain.entity_map)
    model_arguments['relation_count'] = len(ktrain.relation_map)
    scoring_function = getattr(models, model_name)(**model_arguments)
    if has_cuda:
        scoring_function = scoring_function.cuda()
    loss = getattr(losses, loss_function)()
    optim = getattr(torch.optim, optimizer_name)(scoring_function.parameters(), lr=learning_rate)

    if(not eval_batch_size):
        eval_batch_size = max(50, batch_size*2*negative_sample_count//len(ktrain.entity_map))

    if model_name == "image_model":
        tr = trainer.Trainer(scoring_function, scoring_function.regularizer, loss, optim, dltrain, dlvalid, dltest,
                         batch_size=batch_size, eval_batch=eval_batch_size, negative_count=negative_sample_count,
                         save_dir=save_dir, gradient_clip=gradient_clip, hooks=hooks,
                         regularization_coefficient=regularization_coefficient, verbose=verbose, model_name=model_name,
                         image_compatibility = scoring_function.image_compatibility, image_compatibility_coefficient = 0.01)
    else:
        tr = trainer.Trainer(scoring_function, scoring_function.regularizer, loss, optim, dltrain, dlvalid, dltest,
                         batch_size=batch_size, eval_batch=eval_batch_size, negative_count=negative_sample_count,
                         save_dir=save_dir, gradient_clip=gradient_clip, hooks=hooks,
                         regularization_coefficient=regularization_coefficient, verbose=verbose)

    if resume_from_save:
        mb_start = tr.load_state(resume_from_save)
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
    arguments = parser.parse_args()
    if arguments.save_dir is None:
        arguments.save_dir = os.path.join("logs", "%s %s %s run on %s starting from %s" % (arguments.model,
                                                                                        arguments.model_arguments,
                                                                                        arguments.loss,
                                                                                        arguments.dataset,
                                                                                     str(datetime.datetime.now())))
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
    dataset_root = os.path.join(arguments.data_repository_root, arguments.dataset)
    main(dataset_root, arguments.save_dir, arguments.model, arguments.model_arguments, arguments.loss,
         arguments.learning_rate, arguments.batch_size,arguments.regularization_coefficient, arguments.gradient_clip,
         arguments.optimizer, arguments.max_epochs, arguments.negative_sample_count, arguments.hooks,
         arguments.eval_every_x_mini_batches, arguments.eval_batch_size, arguments.resume_from_save,
         arguments.oov_entity, arguments.verbose)
