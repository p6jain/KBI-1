'''
time CUDA_VISIBLE_DEVICES=1 python3 mukund_exp_get_pretrained_scores.py -d fb15k -m image_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}, "image_compatibility_coefficient":1}' -b 6500 -n 200 -v 1 -q 1 -f best_valid_model.pt 

x='-d fb15k -m image_model -a -b 6500 -n 200 -v 1 -q 1 -f best_valid_model.pt';L=x.split(" ")
L = L[:5] + ['{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}, "image_compatibility_coefficient":1}']+L[5:]
import sys
sys.argv += L

'''

import kb
import torch
import os
import csv
import numpy

from collections import defaultdict as dd
def get_entity_mid_name_type_rel_dict(model):
    mid_type=dd(str)
    mid_name=dd(str)
    dataset_root='./data/fb15k/'
    with open("./data/fb15k/entity_mid_name_type_typeid.txt") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            mid_type[row[0]] = row[2]
            mid_name[row[0]] = row[1]
    
    entity_map, type_entity_range = model['entity_map'], model['type_entity_range']
    if 'reverse_entity_map' in model:
        reverse_entity_map = model['reverse_entity_map']
    else:
        reverse_entity_map = {}
        for k,v in entity_map.items():
            reverse_entity_map[v] = k   
    ktrain = kb.kb(os.path.join(dataset_root, 'train.txt'), em=entity_map, type_entity_range=type_entity_range, rem=reverse_entity_map, add_unknowns=True)

    return mid_name,mid_type, ktrain.reverse_relation_map, entity_map, type_entity_range,reverse_entity_map
    
def save_translated_file(file_name="", model=None,data=None):
    mid_name,mid_type,reverse_relation_map,entity_map, type_entity_range,reverse_entity_map = get_entity_mid_name_type_rel_dict(model)
    all_data=dd(list)
    #with open(file_name,"w") as csv_file:
        #csv_writer = csv.writer(csv_file, delimiter=',')    
        #csv_writer.writerow(["r","e1","e2","e1_type","e2_type","e1P","e2P","e1P_type","e2P_type","e1Rank","e2Rank"])
    for r,e1,e2,e1p,e2p,e1r,e2r in data:
            #csv_writer.writerow([reverse_relation_map[int(r)],mid_name[reverse_entity_map[int(e1)]],mid_name[reverse_entity_map[int(e2)]],mid_type[reverse_entity_map[int(e1)]],mid_type[reverse_entity_map[int(e2)]],mid_name[reverse_entity_map[int(e1p)]],mid_name[reverse_entity_map[int(e2p)]],mid_type[reverse_entity_map[int(e1p)]],mid_type[reverse_entity_map[int(e2p)]],e1r,e2r])
        all_data[(e1,r,e2)]=[reverse_relation_map[int(r)], mid_name[reverse_entity_map[int(e1)]],mid_name[reverse_entity_map[int(e2)]],mid_type[reverse_entity_map[int(e1)]],mid_type[reverse_entity_map[int(e2)]],mid_name[reverse_entity_map[int(e1p)]],mid_name[reverse_entity_map[int(e2p)]],mid_type[reverse_entity_map[int(e1p)]],mid_type[reverse_entity_map[int(e2p)]],e1r,e2r]
    return all_data

def save_both_file(model_data, file_name=""):
    with open(file_name,"w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["r","e1","e2","e1_type","e2_type","e1P_typedM","e2P_typedM","e1P_typeM","e2P_typeM","e1Rank_typedM","e2Rank_typedM","e1P_imageM","e2P_imageM","e1P_type_imageM","e2P_type_imageM","e1Rank_imageM","e2Rank_imageM"])
        for e1,r,e2 in model_data["typed_model"].keys():
            csv_writer.writerow(model_data["typed_model"][(e1,r,e2)]+model_data["image_model"][(e1,r,e2)][5:])


def type_analysis(model_data,verbose=1):
    type_performance = {"image_model":dd(int),"typed_model":dd(int),"all_type":dd(int)}
    for tup in model_data["typed_model"].keys():
        r,e1,e2,e1_type,e2_type,e1P_typedM,e2P_typedM,e1P_typeM,e2P_typeM,e1Rank_typedM,e2Rank_typedM,e1P_imageM,e2P_imageM,e1P_type_imageM,e2P_type_imageM,e1Rank_imageM,e2Rank_imageM = model_data["typed_model"][tup]+model_data["image_model"][tup][5:]
        type_performance["all_type"][e1_type]+=1
        type_performance["all_type"][e2_type]+=1
        type_performance["image_model"][e1_type]+=int(e1_type==e1P_type_imageM)
        type_performance["image_model"][e2_type]+=int(e2_type==e2P_type_imageM)
        type_performance["typed_model"][e1_type]+=int(e1_type==e1P_typeM)  
        type_performance["typed_model"][e2_type]+=int(e2_type==e2P_typeM)
        
    for types in type_performance["all_type"].keys():
        type_performance["image_model"][types] = round(100.0*type_performance["image_model"][types]/type_performance["all_type"][types],2)
        type_performance["typed_model"][types] = round(100.0*type_performance["typed_model"][types]/type_performance["all_type"][types],2)
    
    if verbose>0:
        f1 = open("./analysis_fb15k_only_facts_with_image/improved_types.csv",'w')
        f2 = open("./analysis_fb15k_only_facts_with_image/deter_types.csv",'w')
        csv_writer1 = csv.writer(f1,delimiter=',')
        csv_writer2 = csv.writer(f2,delimiter=',')
        improved = [];deter = []
        for types in type_performance["all_type"].keys():
            diff = round(type_performance["image_model"][types] - type_performance["typed_model"][types],2)
            if diff> 10:
                improved.append((types,diff,type_performance["image_model"][types], type_performance["typed_model"][types] ,type_performance["all_type"][types]))
                csv_writer1.writerow([types,diff,type_performance["image_model"][types], type_performance["typed_model"][types] ,type_performance["all_type"][types]])
            elif diff < -10:
                deter.append((types,diff,type_performance["image_model"][types], type_performance["typed_model"][types] ,type_performance["all_type"][types]))
                csv_writer2.writerow([types,diff,type_performance["image_model"][types], type_performance["typed_model"][types] ,type_performance["all_type"][types]])
        print(improved)
        print(deter)
        f1.close()
        f2.close()
    return type_performance

def rel_analysis(model_data,verbose=1):
    r_performance = {"image_model":dd(int),"typed_model":dd(int),"all":dd(int)}
    for tup in model_data["typed_model"].keys():
        r,e1,e2,e1_type,e2_type,e1P_typedM,e2P_typedM,e1P_typeM,e2P_typeM,e1Rank_typedM,e2Rank_typedM,e1P_imageM,e2P_imageM,e1P_type_imageM,e2P_type_imageM,e1Rank_imageM,e2Rank_imageM = model_data["typed_model"][tup]+model_data["image_model"][tup][5:]
        r_performance["all"][r]+=1
        s_t=int(e1_type==e1P_type_imageM)
        o_t=int(e2_type==e2P_type_imageM)
        r_performance["image_model"][r] += int(s_t and o_t)
        s_t=int(e1_type==e1P_typeM)
        o_t=int(e2_type==e2P_typeM)
        r_performance["typed_model"][r] += int(s_t and o_t)

    for r in r_performance["all"].keys():
        r_performance["image_model"][r] = round(100.0*r_performance["image_model"][r]/r_performance["all"][r],2)
        r_performance["typed_model"][r] = round(100.0*r_performance["typed_model"][r]/r_performance["all"][r],2)

    if verbose>0:
        f1 = open("./analysis_fb15k_only_facts_with_image/improved_rels.csv",'w')
        f2 = open("./analysis_fb15k_only_facts_with_image/deter_rels.csv",'w')
        csv_writer1 = csv.writer(f1,delimiter=',')
        csv_writer2 = csv.writer(f2,delimiter=',')
        improved = [];deter = []
        for r in r_performance["all"].keys():
            diff = round(r_performance["image_model"][r] - r_performance["typed_model"][r],2)
            if diff> 10:
                improved.append((r,diff,r_performance["image_model"][r], r_performance["typed_model"][r] ,r_performance["all"][r]))
                csv_writer1.writerow([r,diff,r_performance["image_model"][r], r_performance["typed_model"][r] ,r_performance["all"][r]])
            elif diff < -10:
                deter.append((r,diff,r_performance["image_model"][r], r_performance["typed_model"][r] ,r_performance["all"][r]))
                csv_writer2.writerow([r,diff,r_performance["image_model"][r], r_performance["typed_model"][r] ,r_performance["all"][r]])
        print("improved:")
        for ele in improved:
            print(ele)
        print("deter:")
        for ele in deter:
            print(ele)
        f1.close()
        f2.close()
    return r_performance

def ent_analysis(model_data,verbose=1):
    e_performance = {"image_model":dd(int),"typed_model":dd(int),"all":dd(int)}
    for tup in model_data["typed_model"].keys():
        r,e1,e2,e1_type,e2_type,e1P_typedM,e2P_typedM,e1P_typeM,e2P_typeM,e1Rank_typedM,e2Rank_typedM,e1P_imageM,e2P_imageM,e1P_type_imageM,e2P_type_imageM,e1Rank_imageM,e2Rank_imageM = model_data["typed_model"][tup]+model_data["image_model"][tup][5:]
        e1 = e1.replace(",",":")
        e2 = e2.replace(",",":")
        e_performance["all"][(e1,e1_type)]+=1
        e_performance["all"][(e2,e2_type)]+=1
        s_t=int(e1_type==e1P_type_imageM)
        o_t=int(e2_type==e2P_type_imageM)
        e_performance["image_model"][(e1,e1_type)] += s_t
        e_performance["image_model"][(e2,e2_type)] += o_t
        s_t=int(e1_type==e1P_typeM)
        o_t=int(e2_type==e2P_typeM)
        e_performance["typed_model"][(e1,e1_type)] += s_t
        e_performance["typed_model"][(e2,e2_type)] += o_t 

    for e in e_performance["all"].keys():
        e_performance["image_model"][e] = round(100.0*e_performance["image_model"][e]/e_performance["all"][e],2)
        e_performance["typed_model"][e] = round(100.0*e_performance["typed_model"][e]/e_performance["all"][e],2)

    if verbose>0:
        f1 = open("./analysis_fb15k_only_facts_with_image/improved_ents.csv",'w')
        f2 = open("./analysis_fb15k_only_facts_with_image/deter_ents.csv",'w')
        csv_writer1 = csv.writer(f1,delimiter=',')
        csv_writer2 = csv.writer(f2,delimiter=',')
        improved = [];deter = []
        for e in e_performance["all"].keys():
            diff = round(e_performance["image_model"][e] - e_performance["typed_model"][e],2)
            if diff> 10:
                improved.append((e,diff,e_performance["image_model"][e], e_performance["typed_model"][e] ,e_performance["all"][e]))
                csv_writer1.writerow([e,diff,e_performance["image_model"][e], e_performance["typed_model"][e] ,e_performance["all"][e]])
            elif diff < -10:
                deter.append((e,diff,e_performance["image_model"][e], e_performance["typed_model"][e] ,e_performance["all"][e]))
                csv_writer2.writerow([e,diff,e_performance["image_model"][e], e_performance["typed_model"][e] ,e_performance["all"][e]])
        print("improved:")
        for ele in improved:
            print(ele)
        print("**********************************************************************")
        print("deter:")
        for ele in deter:
            print(ele)
        f1.close()
        f2.close()
    return e_performance

if __name__ == "__main__":
    path = "./analysis_yago/"
    file_name="_analysis_r_e1e2_e1Predictede2Predicted_e1Ranke2Rank.csv"
    model_file="_best_valid_model.pt"
    models=["image_model","typed_model"]
    model_data = {}
    for model_name in models:
        saved_model = torch.load(path+model_name+model_file)
        with open(path+model_name+file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            tmp = save_translated_file(file_name=path+"readable_"+model_name+file_name,model=saved_model,data=csv_reader)
            model_data[model_name] = tmp
    #save_both_file(model_data, file_name=path+"readable_both_"+model_name+file_name)
    print("SAVED COMBINED FILE")
    ta = type_analysis(model_data)
    print("TYPE ANALYSIS DONE")
    ra = rel_analysis(model_data)
    print("REL ANALYSIS DONE")
    ea = ent_analysis(model_data)
    print("ENT ANALYSIS DONE")

