import numpy
import torch
import re
from collections import defaultdict as dd

def get_image_data(dataset_root, flag_dict):
    mid_image =[]
    entity_object_probs = {}
    entity_subject_probs = {}
    relation_image_compat_score = {}

    if flag_dict["flag_use_image"]:#flag_facts_with_image"]:
        #print("Using Facts which have images for both entities")
        mid_image = open(dataset_root+"/mid_image_path.txt").readlines()
        mid_image = set([ele.strip("\n").split("\t")[0] for ele in mid_image])  
        print("num (all inc test/valid) ent w/t image", len(mid_image))

    #ensure we have prob for all ent...if val is user a given bar use 0 or 1 as suited
    if flag_dict["flag_reg_penalty_ent_prob"]:
        print("Penalizing ent_type and image comaptibility scores with entity probability")
        entity_object_probs = open(dataset_root+"/entity_object_probs.csv").readlines()
        entity_subject_probs = open(dataset_root+"/entity_subject_probs.csv").readlines()

        entity_object_probs = dict([((",").join(ele.strip("\n").split(",")[:-1]), float(ele.strip("\n").split(",")[-1])) for ele in entity_object_probs]) 
        entity_subject_probs= dict([((",").join(ele.strip("\n").split(",")[:-1]), float(ele.strip("\n").split(",")[-1])) for ele in entity_subject_probs])
        
    if flag_dict["flag_reg_penalty_image_compat"]:    
        print("Penalizing image_sub and image_obj comaptibility scores with relation scores")
        relation_image_compat_score = open(dataset_root+"/images_large/relation_image_comp_sigmoid.csv").readlines()#relation_mean_image_comp.csv").readlines()
        relation_image_compat_score = dict([((",").join(ele.strip("\n").split(",")[:-2]), float(ele.strip("\n").split(",")[-1])) for ele in relation_image_compat_score]) 

   
    return mid_image, entity_subject_probs, entity_object_probs, relation_image_compat_score    

def convert_to_prob(un_norm_data, total):
    for key in un_norm_data:
        un_norm_data[key] = numpy.sqrt((1.0*un_norm_data[key])/total)
    print("Prob wt: sqrt")
    return un_norm_data

class kb(object):
    """
    Stores a knowledge base as an numpy array. Can be generated from a file. Also stores the entity/relation mappings
    (which is the mapping from entity names to entity id) and possibly entity type information.
    """
    def __init__(self, filename, em=None, rm=None, rem=None, rrm=None, im_em=None, im_rem=None, mid_imid=None, add_unknowns=True, additional_params = {}, nonoov_entity_count=None):
        """
        Duh...
        :param filename: The file name to read the kb from
        :param em: Prebuilt entity map to be used. Can be None for a new map to be created
        :param rm: prebuilt relation map to be used. Same as em
        :param add_unknowns: Whether new entites are to be acknowledged or put as <UNK> token.
        :param additional_params: Support for flag_use_image, 
                                    flag_facts_with_image, flag_reg_penalty_only_images, 
                                    flag_reg_penalty_ent_prob, 
                                    flag_reg_penalty_image_compat (image compatibility is imp only for some relations, 
                                    train set used to get score corresp to every rel - score is avg dot(sub_im, obj_im)) 
        """
        print("Not using image_compat score unless explicitly pecified from now on!")

        self.entity_map = {} if em is None else em
        self.relation_map = {} if rm is None else rm
        self.reverse_entity_map = {} if rem is None else rem
        self.reverse_relation_map = {} if rrm is None else rrm
        
        self.im_entity_map = {} if im_em is None else im_em
        self.im_reverse_entity_map = {} if im_rem is None else im_rem

        self.mid_imid_map = {} if mid_imid is None else mid_imid

        self.entity_id_image_matrix = numpy.array([0])
        self.additional_params = {} if additional_params is None else additional_params

        self.nonoov_entity_count = 0 if nonoov_entity_count is None else nonoov_entity_count #non-oov ent + 1 oov ent
        
        self.entity_prob = {} 
        self.relation_prob = {}
             
        if filename is None:
            return
        facts = []
       
        mid_image = set([]);flag_image = 1
        dataset_root = ("/").join(filename.split("/")[:-1])

        #setting flags here
        if not "flag_use_image" in self.additional_params.keys():
            self.additional_params["flag_use_image"] = 0 
        for key in ["flag_facts_with_image","flag_reg_penalty_only_images","flag_reg_penalty_ent_prob","flag_reg_penalty_image_compat"]:
            if not key in self.additional_params.keys():
                self.additional_params[key] = 0 
            if self.additional_params[key]:
                self.additional_params["flag_use_image"] = 1

        print("additional_params", self.additional_params)
        #get data if resp flag on
        if self.additional_params["flag_use_image"]:
            print("using images")
            mid_image, entity_subject_probs, entity_object_probs, relation_image_compat_score = get_image_data(dataset_root, additional_params)

        with open(filename) as f:
            lines = f.readlines()
            lines = [l.split() for l in lines]

            for l in lines:
                if (not(additional_params['flag_use_image'])) or (not(additional_params['flag_facts_with_image'])) or (additional_params['flag_facts_with_image'] and l[0] in mid_image and l[2] in mid_image):
                    if(add_unknowns):
                        if(l[1] not in self.relation_map):
                            tmp = len(self.relation_map)
                            self.relation_map[l[1]] = tmp
                            self.reverse_relation_map[tmp] = l[1]
                        if(l[0] not in self.entity_map):
                            tmp = len(self.entity_map)
                            self.entity_map[l[0]] = tmp
                            self.reverse_entity_map[tmp] = l[0]
                        if(l[2] not in self.entity_map):
                            tmp = len(self.entity_map)
                            self.entity_map[l[2]] = tmp
                            self.reverse_entity_map[tmp] = l[2]
                    '''
                    if self.additional_param["flag_use_image"]:       
                        if(l[0] not in self.im_entity_map) and (l[0] in mid_image):
                            tmp = len(self.im_entity_map) 
                            self.im_entity_map[l[0]] = tmp
                            self.im_reverse_entity_map[tmp] = l[0]
                        if(l[2] not in self.im_entity_map) and (l[2] in mid_image): 
                            tmp = len(self.im_entity_map)
                            self.im_entity_map[l[2]] = tmp
                            self.im_reverse_entity_map[tmp] = l[2]
                    '''
                    if self.additional_params["flag_use_image"]:#(add_unknowns): #used only for handling neg samples -- data seen in train
                        self.mid_imid_map[self.entity_map.get(l[0], len(self.entity_map)-1)] = self.im_entity_map.get(l[0], self.im_entity_map["<OOV>"])#same::#mid_image))
                        self.mid_imid_map[self.entity_map.get(l[2], len(self.entity_map)-1)] = self.im_entity_map.get(l[2], self.im_entity_map["<OOV>"])#same:#mid_image))  
                        
                    if additional_params['flag_use_image']:
                        entity_reg = []; 
                        image_r_reg = 1.0
                        if (l[0] in mid_image):
                            tmp = entity_subject_probs[l[0]] if (additional_params['flag_reg_penalty_ent_prob'] and l[0] in entity_subject_probs.keys()) else 1.0
                            entity_reg.append(tmp)
                        else:
                            tmp = 0.0 if additional_params['flag_reg_penalty_only_images'] else 1.0
                            entity_reg.append(tmp);

                        if (l[2] in mid_image):
                            tmp = entity_object_probs[l[2]] if (additional_params['flag_reg_penalty_ent_prob'] and l[2] in entity_object_probs.keys()) else 1.0
                            entity_reg.append(tmp);
                        else:
                            tmp = 0.0 if additional_params['flag_reg_penalty_only_images'] else 1.0
                            entity_reg.append(tmp);
                  
                        if l[1] in relation_image_compat_score.keys():
                            image_r_reg = relation_image_compat_score[l[1]] if additional_params['flag_reg_penalty_image_compat'] else 1.0
                        else:
                            image_r_reg = 0.0 #if additional_params['flag_reg_penalty_image_compat'] else 1.0

                        facts.append([self.entity_map.get(l[0], len(self.entity_map)-1), self.relation_map.get(l[1],
                                len(self.relation_map)-1), self.entity_map.get(l[2], len(self.entity_map)-1)] + 
                                [self.im_entity_map.get(l[0], self.im_entity_map["<OOV>"]), self.im_entity_map.get(l[2], self.im_entity_map["<OOV>"])] + 
                                                                                                                               entity_reg + [image_r_reg]) #8
                        # oov id for images = len(mid_image)
                    else:
                        facts.append([self.entity_map.get(l[0], len(self.entity_map)-1), self.relation_map.get(l[1],
                                len(self.relation_map)-1), self.entity_map.get(l[2], len(self.entity_map)-1)])

        self.facts = numpy.array(facts, dtype='int64')

    def augment_image_information(self, mapping):
        """
        Augments the current knowledge base with entity type information for more detailed evaluation.\n
        :param mapping: The maping from entity to types. Expected to be a int to int dict
        :return: None
        """

        self.entity_mid_image_map = mapping
        entity_id_image_map = {}
        for x in self.entity_mid_image_map:
            entity_id_image_map[self.entity_map[x]] = self.entity_mid_image_map[x]
        self.entity_id_image_map = entity_id_image_map#
        size_details = tuple([self.nonoov_entity_count]+list(self.entity_mid_image_map[x].shape[1:]))#len(self.entity_map)]+list(self.entity_mid_image_map[x].shape[1:]))
        entity_id_image_matrix = numpy.zeros(size_details)
        oov_image=numpy.random.rand(1, 3, 224, 224);oov_count=0
        for x in self.entity_map:#self.entity_mid_image_map:
            if x in self.entity_mid_image_map.keys():
                entity_id_image_matrix[self.entity_map[x]] = self.entity_mid_image_map[x]
            else:
                entity_id_image_matrix[self.entity_map[x]] = oov_image
                oov_count+=1
        self.entity_id_image_matrix_np = numpy.array(entity_id_image_matrix, dtype = numpy.long)#
        entity_id_image_matrix = torch.from_numpy(numpy.array(entity_id_image_matrix))
        self.entity_id_image_matrix = entity_id_image_matrix#
    
    def augment_prob_information(self, e_p=None, r_p = None):
        e_prob = dd(int) if e_p is None else e_p
        r_prob = dd(int) if r_p is None else r_p
        if e_p!=None and r_p!=None:
            for e1,r,e2 in self.facts:
                e_prob[e1]+=1; e_prob[e2]+=1
                r_prob[r]+=1
        self.e_prob = convert_to_prob(e_prob, len(self.facts)*2)
        self.r_prob = convert_to_prob(r_prob, len(self.facts))

    def augment_type_information(self, mapping, enm=None, tnm=None):
        """
        Augments the current knowledge base with entity image information.\n
        :param mapping: The maping from entity to types. Expected to be a int to int dict
        :return: None
        """
        self.type_map = mapping
        entity_type_matrix = numpy.zeros((self.nonoov_entity_count, 1))#len(self.entity_map), 1))
        for x in self.type_map:
            entity_type_matrix[self.entity_map[x], 0] = self.type_map[x]
        self.entity_type_matrix_np = numpy.array(entity_type_matrix, dtype = numpy.long)
        entity_type_matrix = torch.from_numpy(numpy.array(entity_type_matrix))
        self.entity_type_matrix = entity_type_matrix

    def compute_degree(self, out=True):
        """
        Computes the in-degree or out-degree of relations\n
        :param out: Whether to compute out-degree or in-degree
        :return: A numpy array with the degree of ith ralation at ith palce.
        """
        entities = [set() for x in self.relation_map]
        index = 2 if out else 0
        for i in range(self.facts.shape[0]):
            entities[self.facts[i][1]].add(self.facts[i][index])
        return numpy.array([len(x) for x in entities])

    def get_id_iid_mapping_all(self, data):
        mapping = self.mid_imid_map
        keys, inv = numpy.unique(data, return_inverse=True)
        vals = numpy.array([mapping[key] for key in keys])
        result = vals[inv]
        return result.reshape(data.shape)




def union(kb_list):
    """
    Computes a union of multiple knowledge bases\n
    :param kb_list: A list of kb
    :return: The union of all kb in kb_list
    """
    l = [k.facts for k in kb_list]
    k = kb(None, kb_list[0].entity_map, kb_list[0].relation_map)
    k.facts = numpy.concatenate(l, 0)
    return k


def dump_mappings(mapping, filename):
    """
    Stores the mapping into a file\n
    :param mapping: The mapping to store
    :param filename: The file name
    :return: None
    """
    data = [[x, mapping[x]] for x in mapping]
    numpy.savetxt(filename, data)


def dump_kb_mappings(kb, kb_name):
    """
    Dumps the entity and relation mapping in a kb\n
    :param kb: The kb
    :param kb_name: The fine name under which the mappings should be stored.
    :return:
    """
    dump_mappings(kb.entity_map, kb_name+".entity")
    dump_mappings(kb.relation_map, kb_name+".relation")
