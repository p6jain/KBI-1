'''
March26'19
Note that the code will not work for case when we allow unk ent (add_unknown=1)
self.kb.nonoov_entity_count is used in place of len(self.kb.entity_map)
'''

import numpy
import torch
import torch.autograd

class data_loader(object):
    """
    Does th job of batching a knowledge base and also generates negative samples with it.
    """
    def __init__(self, kb, load_to_gpu, first_zero=True, flag_add_reverse=None):
        """
        Duh..\n
        :param kb: the knowledge base to batch
        :param load_to_gpu: Whether the batch should be loaded to the gpu or not
        :param first_zero: Whether the first entity in the set of negative samples of each fact should be zero
        """
        self.kb = kb
        self.load_to_gpu = load_to_gpu
        self.first_zero = first_zero
        self.flag_add_reverse = flag_add_reverse

    def get_mapping(self, mapping,data):
        keys, inv = numpy.unique(data, return_inverse=True)
        vals = numpy.array([mapping[key] for key in keys])
        result = vals[inv]
        return result.reshape(data.shape)

    def sample(self, batch_size=1000, negative_count=10):
        """
        Generates a random sample from kb and returns them as numpy arrays.\n
        :param batch_size: The number of facts in the batch or the size of batch.
        :param negative_count: The number of negative samples for each positive fact.
        :return: A list containing s, r, o and negative s and negative o of the batch
        """
        indexes = numpy.random.randint(0, self.kb.facts.shape[0], batch_size)
        facts = self.kb.facts[indexes]
        s = numpy.expand_dims(facts[:, 0], -1)
        r = numpy.expand_dims(facts[:, 1], -1)
        o = numpy.expand_dims(facts[:, 2], -1)

        ns = numpy.random.randint(0, self.kb.nonoov_entity_count, (batch_size, negative_count))
        no = numpy.random.randint(0, self.kb.nonoov_entity_count, (batch_size, negative_count))
        if self.first_zero:
            ns[:, 0] = self.kb.nonoov_entity_count-1
            no[:, 0] = self.kb.nonoov_entity_count-1
        if self.kb.additional_params["flag_use_image"]:#self.kb.use_image:
            #s_image = numpy.array(self.kb.entity_id_image_matrix[s]).squeeze(1)
            #o_image = numpy.array(self.kb.entity_id_image_matrix[o]).squeeze(1)
            #return [s, r, o, ns, no, s_image, o_image]
            s_im = numpy.expand_dims(facts[:, 3], -1)
            o_im = numpy.expand_dims(facts[:, 4], -1)
            s_oov = numpy.expand_dims(facts[:, 5], -1)
            o_oov = numpy.expand_dims(facts[:, 6], -1)
            s_oov = s_oov.astype(float)
            o_oov = o_oov.astype(float)
            ic_r_score = numpy.expand_dims(facts[:, 7], -1)
            ic_r_score = ic_r_score.astype(float)

            ns_im = self.get_mapping(self.kb.mid_imid_map, ns) ##handle mid to image-id mapping here!!!
            no_im = self.get_mapping(self.kb.mid_imid_map, no)
            return [s, r, o, ns, no, s_im, o_im, ns_im, no_im, s_oov, o_oov, ic_r_score]
        else:
            return [s, r, o, ns, no]

    def sample_icml(self, batch_size=1000, negative_count=10):
        """
        Generates a random sample from kb and returns them as numpy arrays.\n
        :param batch_size: The number of facts in the batch or the size of batch.
        :param negative_count: The number of negative samples for each positive fact.
        :return: A list containing s, r, o and negative s and negative o of the batch
        """
        indexes = numpy.random.randint(0, self.kb.facts.shape[0]*2 , batch_size)
        indexes = indexes.reshape((indexes.shape[0],-1))
        check = indexes > (self.kb.facts.shape[0] - 1); check.astype('int');
        check=check.reshape((check.shape[0],-1))
        new_indexes = indexes - (check * self.kb.facts.shape[0])
        new_indexes = new_indexes.reshape((new_indexes.shape[0],))
        facts = self.kb.facts[new_indexes]

        s_tmp = numpy.expand_dims(facts[:, 0], -1)
        r_tmp = numpy.expand_dims(facts[:, 1], -1)
        o_tmp = numpy.expand_dims(facts[:, 2], -1)

        num_relations = len(self.kb.relation_map)
        tmp_r_change = check*num_relations
        r = r_tmp + check * (num_relations) 
        s, o = numpy.where(check, (o_tmp, s_tmp), (s_tmp, o_tmp))        

        ns = numpy.random.randint(0, self.kb.nonoov_entity_count, (batch_size, negative_count))
        no = numpy.random.randint(0, self.kb.nonoov_entity_count, (batch_size, negative_count))
        if self.first_zero:
            ns[:, 0] = len(self.kb.nonoov_entity_count)-1
            no[:, 0] = len(self.kb.nonoov_entity_count)-1

        if 0:#self.kb.entity_id_image_matrix.shape[0] > 1:
            s_image = numpy.array(self.kb.entity_id_image_matrix[s]).squeeze(1)
            o_image = numpy.array(self.kb.entity_id_image_matrix[o]).squeeze(1)
            return [s, r, o, ns, no, s_image, o_image]
        else:
            return [s, r, o, ns, no]


    def sample_outside_type_range(self, type_exclude, negative_count):
        start, end = self.kb.type_entity_range[type_exclude] #
        #start = end = int(len(self.kb.entity_map)/2.0)
        diff1 = start
        diff2 = self.kb.nonoov_entity_count - end - 1 #
        ratio = 1.0*(diff1)/(diff2+diff1)
        if int(ratio * negative_count) >= 1 and (negative_count - int(ratio * negative_count)) > 1:
            a1 = numpy.random.choice(start, int(ratio * negative_count))
            a2 = numpy.random.choice(self.kb.nonoov_entity_count - end - 1, negative_count - int(ratio * negative_count)) + end + 1
        elif (negative_count - int(ratio * negative_count)) > 1 and int(ratio * negative_count) < 1 :
            a1 = numpy.array([])
            a2 = numpy.random.choice(self.kb.nonoov_entity_count - end - 1, negative_count) + end + 1
        else:
            a1 = numpy.random.choice(start,  negative_count)
            a2 = numpy.array([])

        #a1 = numpy.random.randint(0, start, int(diff1 * ratio))
        #diff = len(self.kb.entity_map) - start - 1
        #a2 = numpy.random.randint(end+1, len(self.kb.entity_map), int(diff2 * (1-ratio)))
        a = numpy.concatenate((a1,a2), axis = 0)#[numpy.random.choice(a1.shape[0]+a2.shape[0], negative_count, replace=False)]
        assert a.shape[0] == negative_count
        return a

    def sample_inside_type_range(self, type_include, negative_count):
        start, end = self.kb.type_entity_range[type_include]
        a = numpy.random.choice((end-start)+1, negative_count) + start
        return a


    def sample_neg_sensitive(self, batch_size=1000, negative_count=10):
        indexes = numpy.random.randint(0, self.kb.facts.shape[0], batch_size)
        facts = self.kb.facts[indexes]
        s = numpy.expand_dims(facts[:, 0], -1)
        r = numpy.expand_dims(facts[:, 1], -1)
        o = numpy.expand_dims(facts[:, 2], -1)
        ns = []; no = []
        ns2 = []; no2 = []
        type_s = self.kb.entity_type_matrix_np[s]
        type_o = self.kb.entity_type_matrix_np[o]

        for i in range(s.shape[0]):
            ns.append(self.sample_outside_type_range(type_s[i][0][0], negative_count))
            no.append(self.sample_outside_type_range(type_o[i][0][0], negative_count))
            ns2.append(self.sample_inside_type_range(type_s[i][0][0], int(negative_count/10)))
            no2.append(self.sample_inside_type_range(type_o[i][0][0], int(negative_count/10)))

        ns = numpy.array(ns, dtype = numpy.long)
        no = numpy.array(no, dtype = numpy.long)
        ns2 = numpy.array(ns2, dtype = numpy.long)
        no2 = numpy.array(no2, dtype = numpy.long)

        return [s, r, o, ns, no, ns2, no2]

    def tensor_sample(self, batch_size=1000, negative_count=10):
        """
        Generates a random sample from kb and returns them as torch tensors. Internally uses sampe\n
        :param batch_size: The number of facts in the batch or the size of batch.
        :param negative_count: The number of negative samples for each positive fact.
        :return: A list containing s, r, o and negative s and negative o of the batch
        """
        # ls = self.sample_neg_sensitive(batch_size, negative_count)
        if self.flag_add_reverse:
            ls = self.sample_icml(batch_size, negative_count)
        else:
            ls = self.sample(batch_size, negative_count)
        if self.load_to_gpu:
            return [torch.autograd.Variable(torch.from_numpy(x).cuda()) for x in ls]
        else:
            return [torch.autograd.Variable(torch.from_numpy(x)) for x in ls]
