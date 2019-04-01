import numpy
import torch
import re

class kb(object):
    """
    Stores a knowledge base as an numpy array. Can be generated from a file. Also stores the entity/relation mappings
    (which is the mapping from entity names to entity id) and possibly entity type information.
    """
    def __init__(self, filename, em=None, rm=None, type_entity_range=None, rem=None, rrm=None, add_unknowns=True, use_image=0):
        """
        Duh...
        :param filename: The file name to read the kb from
        :param em: Prebuilt entity map to be used. Can be None for a new map to be created
        :param rm: prebuilt relation map to be used. Same as em
        :param add_unknowns: Whether new entites are to be acknowledged or put as <UNK> token.
        """
        self.entity_map = {} if em is None else em
        self.relation_map = {} if rm is None else rm
        self.type_entity_range = {} if type_entity_range is None else type_entity_range
        self.reverse_entity_map = {} if rem is None else rem
        self.reverse_relation_map = {} if rrm is None else rrm
        self.entity_id_image_matrix = numpy.array([0])
        self.use_image = use_image
        if filename is None:
            return
        facts = []

        self.entity_sub_weight = {} 
        self.entity_obj_weight = {}
        
        '''
        tmp: removing facts with no image
        '''
        flag_selective_reg = 0
        flag_use_prob = 0 #harm
        flag_use_image_compat_score = 0
        if flag_use_prob:
            print("using entity prob as reg coeff")

        if flag_selective_reg:
            print("not reg entities w/o image")

        if flag_use_image_compat_score:
            print("using rel scores as image compatibility score reg coef!")

        mid_image = set([]);flag_image = 1
        dataset_root = ("/").join(filename.split("/")[:-1])
        if self.use_image:
            print("using images")
            #print("removing facts with no image!!")
            mid_image = open(dataset_root+"/mid_image_path.txt").readlines()
            mid_image = set([ele.strip("\n").split("\t")[0] for ele in mid_image])
            flag_image = 0
            print("size of mid_image", len(mid_image))
           
            entity_object_probs = {}
            entity_subject_probs = {}
            relation_image_compatibility_support_score = {}
 
            if 0:
                entity_object_probs = open(dataset_root+"/entity_object_probs.csv").readlines()
                entity_subject_probs = open(dataset_root+"/entity_subject_probs.csv").readlines()

                entity_object_probs = dict([((",").join(ele.strip("\n").split(",")[:-1]), float(ele.strip("\n").split(",")[-1])) for ele in entity_object_probs]) 
                entity_subject_probs= dict([((",").join(ele.strip("\n").split(",")[:-1]), float(ele.strip("\n").split(",")[-1])) for ele in entity_subject_probs])
            
                relation_image_compatibility_support_score = open(dataset_root+"/images_large/relation_image_comp_sigmoid.csv").readlines()#relation_mean_image_comp.csv").readlines()
                relation_image_compatibility_support_score = dict([((",").join(ele.strip("\n").split(",")[:-2]), float(ele.strip("\n").split(",")[-1])) for ele in relation_image_compatibility_support_score]) 

        rem=0
        with open(filename) as f:
            lines = f.readlines()
            lines = [l.split() for l in lines]

            for l in lines:
                if 1:##flag_image or (l[0] in mid_image and l[2] in mid_image):
                    if(add_unknowns):
                        if(l[1] not in self.relation_map):
                            self.reverse_relation_map[len(self.relation_map)] = l[1]
                            self.relation_map[l[1]] = len(self.relation_map)
                        if(l[0] not in self.entity_map):
                            self.reverse_entity_map[len(self.entity_map)] = l[0]
                            self.entity_map[l[0]] = len(self.entity_map)
                        if(l[2] not in self.entity_map):
                            self.reverse_entity_map[len(self.entity_map)] = l[2]
                            self.entity_map[l[2]] = len(self.entity_map)
                #only for all facts
                if self.use_image:
                    L=[]; 
                    R=[]
                    K=[]#store info if facts has image or not
                    if (l[0] in mid_image):
                        tmp = entity_subject_probs[l[0]] if (flag_use_prob and l[0] in entity_subject_probs.keys()) else 1.0
                        L.append(tmp)
                        K.append(1)
                    else:
                        tmp = 0.0 if flag_selective_reg else 1.0
                        L.append(tmp);K.append(0)
                    if (l[2] in mid_image):
                        tmp = entity_object_probs[l[2]] if (flag_use_prob and l[2] in entity_object_probs.keys()) else 1.0
                        L.append(tmp);K.append(1)
                    else:
                        tmp = 0.0 if flag_selective_reg else 1.0
                        L.append(tmp);K.append(1)
                  
                    if l[1] in relation_image_compatibility_support_score.keys():
                        R = relation_image_compatibility_support_score[l[1]] if flag_use_image_compat_score else 1.0
                    else:
                        R = 0.0 if flag_use_image_compat_score else 1.0

                    #if K[0]+K[1] < 2:
                    #    rem=1;continue;
                        #print("removing facts with no image!!");continue
                    #L = [1.0, 1.0]; R = 1.0##old image model -- all on 
                    facts.append([self.entity_map.get(l[0], len(self.entity_map)-1), self.relation_map.get(l[1],
                                len(self.relation_map)-1), self.entity_map.get(l[2], len(self.entity_map)-1)]+L+[R])
                else:
                    facts.append([self.entity_map.get(l[0], len(self.entity_map)-1), self.relation_map.get(l[1],
                                len(self.relation_map)-1), self.entity_map.get(l[2], len(self.entity_map)-1)])
                #
            if rem:
                print("########removing facts with no image!!#######")
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
        size_details = tuple([len(self.entity_map)]+list(self.entity_mid_image_map[x].shape[1:]))
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


    def augment_type_information(self, mapping, enm=None, tnm=None):
        """
        Augments the current knowledge base with entity image information.\n
        :param mapping: The maping from entity to types. Expected to be a int to int dict
        :return: None
        """
        self.type_map = mapping
        entity_type_matrix = numpy.zeros((len(self.entity_map), 1))
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
