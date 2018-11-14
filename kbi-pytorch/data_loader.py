import numpy
import torch
import torch.autograd


class data_loader(object):
    """
    Does th job of batching a knowledge base and also generates negative samples with it.
    """
    def __init__(self, kb, load_to_gpu, first_zero=True):
        """
        Duh..\n
        :param kb: the knowledge base to batch
        :param load_to_gpu: Whether the batch should be loaded to the gpu or not
        :param first_zero: Whether the first entity in the set of negative samples of each fact should be zero
        """
        self.kb = kb
        self.load_to_gpu = load_to_gpu
        self.first_zero = first_zero

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
        ns = numpy.random.randint(0, len(self.kb.entity_map), (batch_size, negative_count))
        no = numpy.random.randint(0, len(self.kb.entity_map), (batch_size, negative_count))
        if self.first_zero:
            ns[:, 0] = len(self.kb.entity_map)-1
            no[:, 0] = len(self.kb.entity_map)-1
        return [s, r, o, ns, no]

    def fun(self, type_exclude, negative_count):
        start, end = self.kb.type_entity_range[type_exclude]; # 
        start = end = int(len(self.kb.entity_map)/2.0)
        diff1 = start 
        diff2 = len(self.kb.entity_map) - end + 1 #
        ratio = 1.0*(diff1)/(diff2+diff1)
        if int(ratio * negative_count) >= 1 and (negative_count - int(ratio * negative_count)) > 1:
            a1 = numpy.random.choice(start, int(ratio * negative_count))
            a2 = numpy.random.choice(len(self.kb.entity_map) - end - 1, negative_count - int(ratio * negative_count)) + end + 1
        elif (negative_count - int(ratio * negative_count)) > 1 and int(ratio * negative_count) < 1 :
            a1 = numpy.array([])
            a2 = numpy.random.choice(len(self.kb.entity_map) - end - 1, negative_count) + end + 1
        else:
            a1 = numpy.random.choice(start,  negative_count)
            a2 = numpy.array([])

        #a1 = numpy.random.randint(0, start, int(diff1 * ratio))
        #diff = len(self.kb.entity_map) - start - 1
        #a2 = numpy.random.randint(end+1, len(self.kb.entity_map), int(diff2 * (1-ratio)))
        a = numpy.concatenate((a1,a2), axis = 0)#[numpy.random.choice(a1.shape[0]+a2.shape[0], negative_count, replace=False)]
        assert a.shape[0] == negative_count
        return a

    def sample_neg_sensitive(self, batch_size=1000, negative_count=10):
        indexes = numpy.random.randint(0, self.kb.facts.shape[0], batch_size)
        facts = self.kb.facts[indexes]
        s = numpy.expand_dims(facts[:, 0], -1)
        r = numpy.expand_dims(facts[:, 1], -1)
        o = numpy.expand_dims(facts[:, 2], -1)
        ns = []; no = []
        type_s = self.kb.entity_type_matrix_np[s]
        type_o = self.kb.entity_type_matrix_np[o]

        # print(type_s.shape)
        # print(type_s)

        for i in range(s.shape[0]):
            ns.append(self.fun(type_s[i][0][0], negative_count))
            no.append(self.fun(type_o[i][0][0], negative_count))
        ns = numpy.array(ns, dtype = numpy.long)
        no = numpy.array(no, dtype = numpy.long)

        # print(ns.shape)
        # print(ns)
        # print(ns.shape)

        return [s, r, o, ns, no]

    def tensor_sample(self, batch_size=1000, negative_count=10):
        """
        Generates a random sample from kb and returns them as torch tensors. Internally uses sampe\n
        :param batch_size: The number of facts in the batch or the size of batch.
        :param negative_count: The number of negative samples for each positive fact.
        :return: A list containing s, r, o and negative s and negative o of the batch
        """
        ls = self.sample_neg_sensitive(batch_size, negative_count)#self.sample(batch_size, negative_count)
        if self.load_to_gpu:
            return [torch.autograd.Variable(torch.from_numpy(x).cuda()) for x in ls]
        else:
            return [torch.autograd.Variable(torch.from_numpy(x)) for x in ls]
