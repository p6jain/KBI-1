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
        
        '''
        tmp: removing facts with no image
        '''
        mid_image = set([]);flag_image = 1
        dataset_root = ("/").join(filename.split("/")[:-1])
        if self.use_image:
            print("using images")
            #print("removing facts with no image!!")
            mid_image = open(dataset_root+"/mid_image_path.txt").readlines()
            mid_image = set([ele.strip("\n").split("\t")[0] for ele in mid_image])
            flag_image = 0
            print("size of mid_image", len(mid_image))
        rem=0
        with open(filename) as f:
            lines = f.readlines()
            lines = [l.split() for l in lines]

            for l in lines:
                if self.use_image:
                    L=[]
                    if (l[0] in mid_image):
                        L.append(1.0)
                    else:
                        L.append(0.0)
                    if (l[2] in mid_image):
                        L.append(1.0)
                    else:
                        L.append(0.0)
                   
                    if L[0]+L[1] < 2:
                        rem=1;continue;
 
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
                    L = [1.0, 1.0]##old image model -- all on 
                    facts.append([self.entity_map.get(l[0], len(self.entity_map)-1), self.relation_map.get(l[1],
                                len(self.relation_map)-1), self.entity_map.get(l[2], len(self.entity_map)-1)]+L)
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
        #print("Prachi","kb","1")
        for x in self.entity_mid_image_map:
            entity_id_image_map[self.entity_map[x]] = self.entity_mid_image_map[x]
        #print("Prachi","kb","2")
        self.entity_id_image_map = entity_id_image_map#
        #print("Prachi","kb","3")
        #size_details = tuple([len(self.entity_mid_image_map)]+list(self.entity_mid_image_map[x].shape[1:]))
        size_details = tuple([len(self.entity_map)]+list(self.entity_mid_image_map[x].shape[1:]))
        entity_id_image_matrix = numpy.zeros(size_details)
        #print("Prachi","kb","4")
        oov_image=numpy.random.rand(1, 3, 224, 224);oov_count=0
        for x in self.entity_map:#self.entity_mid_image_map:
            if x in self.entity_mid_image_map.keys():
                entity_id_image_matrix[self.entity_map[x]] = self.entity_mid_image_map[x]
            else:
                entity_id_image_matrix[self.entity_map[x]] = oov_image
                oov_count+=1
        #print("Prachi","kb","5", oov_count)
        self.entity_id_image_matrix_np = numpy.array(entity_id_image_matrix, dtype = numpy.long)#
        #print("Prachi","kb","6")
        entity_id_image_matrix = torch.from_numpy(numpy.array(entity_id_image_matrix))
        #print("Prachi","kb","7")
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
