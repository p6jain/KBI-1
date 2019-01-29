import torch
import utils
import random
import torchvision.models as torchvision_models
import torch.nn as nn
import torch.nn.functional as F

class distmult(torch.nn.Module):
    """
    DistMult Model from Trullion et al 2014.\n
    Scoring function (s, r, o) = <s, r, o> # dot product
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False, reg=2):
        """
        The initializing function. These parameters are expected to be supplied from the command line when running the\n
        program from main.\n
        :param entity_count: The number of entities in the knowledge base/model
        :param relation_count: Number of relations in the knowledge base/model
        :param embedding_dim: The size of the embeddings of entities and relations
        :param unit_reg: Whether the ___entity___ embeddings should be unit regularized or not
        :param clamp_v: The value at which to clamp the scores. (necessary to avoid over/underflow with some losses
        :param display_norms: Whether to display the max and min entity and relation embedding norms with each update
        :param reg: The type of regularization (example-l1,l2)
        """
        super(distmult, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.reg = reg

        self.display_norms = display_norms
        self.E = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v

    def forward(self, s, r, o, flag_debug=0):
        """
        This is the scoring function \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the scoring function
        """
        s = self.E(s) if s is not None else self.E.weight.unsqueeze(0)
        r = self.R(r)
        o = self.E(o) if o is not None else self.E.weight.unsqueeze(0)
        if self.clamp_v:
            s.data.clamp_(-self.clamp_v, self.clamp_v)
            r.data.clamp_(-self.clamp_v, self.clamp_v)
            o.data.clamp_(-self.clamp_v, self.clamp_v)
        return (s*r*o).sum(dim=-1)

    def regularizer(self, s, r, o):
        """
        This is the regularization term \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the regularization term
        """
        s = self.E(s)
        r = self.R(r)
        o = self.E(o)
        if self.reg==2:
            return (s*s+o*o+r*r).sum()
        elif self.reg == 1:
            return (s.abs()+r.abs()+o.abs()).sum()
        else:
            print("Unknown reg for distmult model")
            assert(False)

    def post_epoch(self):
        """
        Post epoch/batch processing stuff.
        :return: Any message that needs to be displayed after each batch
        """
        if (not self.unit_reg and not self.display_norms):
            return ""
        e_norms = self.E.weight.data.norm(2, dim=-1, keepdim=True)
        r_norms = self.R.weight.data.norm(2, dim=-1)
        max_e, min_e = torch.max(e_norms), torch.min(e_norms)
        max_r, min_r = torch.max(r_norms), torch.min(r_norms)
        if self.unit_reg:
            self.E.weight.data.div_(e_norms)
        if self.display_norms:
            return "E[%4f, %4f] R[%4f, %4f]" % (max_e, min_e, max_r, min_r)
        else:
            return ""


class complex(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, clamp_v=None):
        super(complex, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E_im = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_im = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.E_re = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_re = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v

    def forward(self, s, r, o, flag_debug=0):
        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)
        if self.clamp_v:
            s_im.data.clamp_(-self.clamp_v, self.clamp_v)
            s_re.data.clamp_(-self.clamp_v, self.clamp_v)
            r_im.data.clamp_(-self.clamp_v, self.clamp_v)
            r_re.data.clamp_(-self.clamp_v, self.clamp_v)
            o_im.data.clamp_(-self.clamp_v, self.clamp_v)
            o_re.data.clamp_(-self.clamp_v, self.clamp_v)
        result = (s_re*o_re+s_im*o_im)*r_re + (s_re*o_im-s_im*o_re)*r_im
        return result.sum(dim=-1)

    def regularizer(self, s, r, o):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        return (s_re*s_re+o_re*o_re+r_re*r_re+s_im*s_im+r_im*r_im+o_im*o_im).sum()

    def post_epoch(self):
        return ""


class adder_model(torch.nn.Module):
    def __init__(self, entity_count, relation_count, model1_name, model1_arguments, model2_name, model2_arguments):
        super(adder_model, self).__init__()

        model1 = globals()[model1_name]
        model1_arguments['entity_count'] = entity_count
        model1_arguments['relation_count'] = relation_count
        model2 = globals()[model2_name]
        model2_arguments['entity_count'] = entity_count
        model2_arguments['relation_count'] = relation_count

        self.model1 = model1(**model1_arguments)
        self.model2 = model2(**model2_arguments)
        self.minimum_value = self.model1.minimum_value + self.model2.minimum_value

    def forward(self, s, r, o, flag_debug=0):
        return self.model1(s, r, o) + self.model2(s, r, o)

    def post_epoch(self):
        return self.model1.post_epoch()+self.model2.post_epoch()

    def regularizer(self, s, r, o):
        return self.model1.regularizer(s, r, o) + self.model2.regularizer(s, r, o)


class typed_model(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0):
        super(typed_model, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.mult = mult
        self.psi = psi
        self.E_t = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_ht = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tt = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        self.minimum_value = 0.0

    def forward(self, s, r, o, flag_debug=0):
        base_forward = self.base_model(s, r, o)
        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)
        head_type_compatibility = (s_t*r_ht).sum(-1)
        tail_type_compatibility = (o_t*r_tt).sum(-1)
        base_forward = torch.nn.Sigmoid()(self.psi*base_forward)
        head_type_compatibility = torch.nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = torch.nn.Sigmoid()(self.psi*tail_type_compatibility)
        return self.mult*base_forward*head_type_compatibility*tail_type_compatibility #, base_forward, head_type_compatibility, tail_type_compatibility

    def regularizer(self, s, r, o):
        """
        s_t = self.E_t(s)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o)
        reg = (s_t*s_t + r_ht*r_ht + r_tt*r_tt + o_t*o_t).sum()
        return self.base_model.regularizer(s, r, o) + reg
        """
        return self.base_model.regularizer(s, r, o)

    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
        return self.base_model.post_epoch()

class EncoderCNN(torch.nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = torchvision_models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images, flag_debug=0):
        """Extract feature vectors from input images."""
        images = images.float()
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class image_model(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, image_embedding=None):
        super(image_model, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.mult = mult
        self.psi = psi
        self.E_t = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_ht = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tt = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        self.minimum_value = 0.0

        #image model
        #self.image_model = EncoderCNN(self.embedding_dim)
        self.image_embedding = torch.from_numpy(image_embedding).to('cuda')

    def forward(self, s, r, o, flag_debug=0):
        base_forward = self.base_model(s, r, o)
        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)
        head_type_compatibility = (s_t*r_ht).sum(-1)
        #print("Prachi Debug","s_t,r_ht", s_t.shape, r_ht.shape)
        tail_type_compatibility = (o_t*r_tt).sum(-1)
        base_forward = torch.nn.Sigmoid()(self.psi*base_forward)
        head_type_compatibility = torch.nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = torch.nn.Sigmoid()(self.psi*tail_type_compatibility)

        tmp = self.mult*base_forward*head_type_compatibility*tail_type_compatibility
        #print("Prachi Debug","final type model score",tmp.shape)
        return self.mult*base_forward*head_type_compatibility*tail_type_compatibility #, base_forward, head_type_compatibility, tail_type_compatibility

    def regularizer(self, s, r, o):
        """
        s_t = self.E_t(s)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o)
        reg = (s_t*s_t + r_ht*r_ht + r_tt*r_tt + o_t*o_t).sum()
        return self.base_model.regularizer(s, r, o) + reg
        """
        return self.base_model.regularizer(s, r, o)

    def image_compatibility(self, s, o):
        '''
        need to write a seperate function for evaluation scores
        -- as we plan to prestore all entity's image embeddings
        '''
        #print("Prachi Debug", s.shape, s_image.shape, o.shape, o_image.shape)
        s_t = self.E_t(s) #if s is not None else self.E_t.weight.unsqueeze(0)
        o_t = self.E_t(o) #if o is not None else self.E_t.weight.unsqueeze(0)

        s_image = self.image_embedding(s)
        o_image = self.image_embedding(o)

        #print("Prachi Debug","s_image.shape",s_image.shape)
        #print("Prachi Debug","s_t.shape",s_t.shape)
        s_image.unsqueeze_(1)
        o_image.unsqueeze_(1)

        #print("Prachi Debug","s_image.shape 2",s_image.shape)

        s_image_compatibility = (s_t * s_image).sum(-1)
        o_image_compatibility = (o_t * o_image).sum(-1)
        #print("Prachi Debug","s_image_compatibility.shape",s_image_compatibility.shape)
        ##
        s_image_compatibility = torch.nn.Sigmoid()(self.psi*s_image_compatibility)
        o_image_compatibility = torch.nn.Sigmoid()(self.psi*o_image_compatibility)
        #print("Prachi Debug","s_image_compatibility.shape", s_image_compatibility.shape)

        s_image_compatibility.squeeze()
        o_image_compatibility.squeeze()

        #print("Prachi Debug","s_image_compatibility.shape 2", s_image_compatibility.shape)

        return -(torch.mean(s_image_compatibility)), -(torch.mean(o_image_compatibility))

    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
        return self.base_model.post_epoch()


class DME(torch.nn.Module):
    """
    DM+E model.
    deprecated. Use Adder model with DM and E as sub models for more control
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False):
        super(DME, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg

        self.E_DM = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_DM = torch.nn.Embedding(self.relation_count, self.embedding_dim)

        torch.nn.init.normal_(self.E_DM.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_DM.weight.data, 0, 0.05)

        self.E = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_head = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tail = torch.nn.Embedding(self.relation_count, self.embedding_dim)

        torch.nn.init.normal_(self.E.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_head.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tail.weight.data, 0, 0.05)

        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v
        self.diplay_norms=display_norms

    def forward(self, s, r, o, flag_debug=0):
        s_DM = self.E_DM(s) if s is not None else self.E_DM.weight.unsqueeze(0)
        r_DM = self.R_DM(r)
        o_DM = self.E_DM(o) if o is not None else self.E_DM.weight.unsqueeze(0)

        s = self.E(s) if s is not None else self.E.weight.unsqueeze(0)
        r_head = self.R_head(r)
        r_tail = self.R_tail(r)
        o = self.E(o) if o is not None else self.E.weight.unsqueeze(0)

        if self.clamp_v:
            s.data.clamp_(-self.clamp_v, self.clamp_v)
            r_head.data.clamp_(-self.clamp_v, self.clamp_v)
            r_tail.data.clamp_(-self.clamp_v, self.clamp_v)
            o.data.clamp_(-self.clamp_v, self.clamp_v)

            s_DM.data.clamp_(-self.clamp_v, self.clamp_v)
            r_DM.data.clamp_(-self.clamp_v, self.clamp_v)

        out = (s*r_head+o*r_tail).sum(dim=-1) + (s_DM*r_DM*o_DM).sum(dim=-1)
        return out

    def regularizer(self, s, r, o):
        s_DM = self.E_DM(s)
        r_DM = self.R_DM(r)
        o_DM = self.E_DM(o)

        s = self.E(s)
        r_head = self.R_head(r)
        r_tail = self.R_tail(r)
        o = self.E(o)

        return (s*s+o*o+r_head*r_head+r_tail*r_tail+s_DM*s_DM+r_DM*r_DM+o_DM*o_DM).sum()#(s*s+o*o+r*r).sum()

    def post_epoch(self):
        e_norms = self.E.weight.data.norm(2, dim=-1, keepdim=True)
        r_head_norms = self.R_head.weight.data.norm(2, dim=-1)
        r_tail_norms = self.R_tail.weight.data.norm(2, dim=-1)

        max_e, min_e = torch.max(e_norms), torch.min(e_norms)
        max_r_tail, min_r_tail = torch.max(r_tail_norms), torch.min(r_tail_norms)
        max_r_head, min_r_head = torch.max(r_head_norms), torch.min(r_head_norms)

        e_DM_norms = self.E_DM.weight.data.norm(2, dim=-1, keepdim=True)
        r_DM_norms = self.R_DM.weight.data.norm(2, dim=-1)

        max_e_DM, min_e_DM = torch.max(e_DM_norms), torch.min(e_DM_norms)
        max_r_DM, min_r_DM = torch.max(r_DM_norms), torch.min(r_DM_norms)

        if self.unit_reg:
            self.E.weight.data.div_(e_norms)
            self.E_DM.weight.data.div_(e_DM_norms)
        if self.diplay_norms:
            return "(%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f)" % (max_e, min_e, max_r_head, min_r_head, max_r_tail, min_r_tail, max_e_DM, min_e_DM, max_r_DM, min_r_DM)
        else:
            return ""


class E(torch.nn.Module):
    """
    E model \n
    scoring function (s, r, o) = s*r_h + o*r_o
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=True, clamp_v=None, display_norms=False):
        super(E, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg

        self.E = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_head = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tail = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_head.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tail.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v
        self.display_norms = display_norms

    def forward(self, s, r, o, flag_debug=0):
        s = self.E(s) if s is not None else self.E.weight.unsqueeze(0)
        r_head = self.R_head(r)
        r_tail = self.R_tail(r)
        o = self.E(o) if o is not None else self.E.weight.unsqueeze(0)
        if self.clamp_v:
            s.data.clamp_(-self.clamp_v, self.clamp_v)
            r_head.data.clamp_(-self.clamp_v, self.clamp_v)
            r_tail.data.clamp_(-self.clamp_v, self.clamp_v)
            o.data.clamp_(-self.clamp_v, self.clamp_v)
        return (s*r_head+o*r_tail).sum(dim=-1)

    def regularizer(self, s, r, o):
        s = self.E(s)
        r_head = self.R_head(r)
        r_tail = self.R_tail(r)
        o = self.E(o)
        return (s*s+o*o+r_head*r_head+r_tail*r_tail).sum()#(s*s+o*o+r*r).sum()

    def post_epoch(self):
        e_norms = self.E.weight.data.norm(2, dim=-1, keepdim=True)
        r_head_norms = self.R_head.weight.data.norm(2, dim=-1)
        r_tail_norms = self.R_tail.weight.data.norm(2, dim=-1)

        max_e, min_e = torch.max(e_norms), torch.min(e_norms)
        max_r_tail, min_r_tail = torch.max(r_tail_norms), torch.min(r_tail_norms)
        max_r_head, min_r_head = torch.max(r_head_norms), torch.min(r_head_norms)

        if self.unit_reg:
            self.E.weight.data.div_(e_norms)
        if self.display_norms:
            return "(%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f)" % (max_e, min_e, max_r_head, min_r_head, max_r_tail, min_r_tail)
        else:
            return ""

def print_fun(data):
    print("Prachi Analysis::", data)

class box_typed_model(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, mult=20.0, box_reg_coef=0.1, box_reg='l2', psi=2.0):
        super(box_typed_model, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.mult = mult
        self.box_reg = box_reg
        self.box_reg_coef = box_reg_coef
        self.psi = psi
        self.E_t = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_ht_high = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_ht_low = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tt_high = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tt_low = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht_high.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt_low.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt_high.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht_low.weight.data, 0, 0.05)
        #ensuring _low _high are init appropriately..low is low and vice versa
        self.R_ht_high.weight.data, self.R_ht_low.weight.data = torch.max(self.R_ht_high.weight.data, self.R_ht_low.weight.data), torch.min(self.R_ht_high.weight.data, self.R_ht_low.weight.data)
        self.R_tt_high.weight.data, self.R_tt_low.weight.data = torch.max(self.R_tt_high.weight.data, self.R_tt_low.weight.data), torch.min(self.R_tt_high.weight.data, self.R_tt_low.weight.data)

        self.minimum_value = 0.0

    def compute_distance(self, box_low, box_high, point):
        delta = 0.001
        temporary = (box_low + delta) - point
        term_1 = torch.max(temporary, torch.zeros(1).cuda() if temporary.is_cuda else torch.zeros(1))
        distance = torch.max(point - (box_high + delta), term_1)
        distance, _ = distance.max(dim=-1)
        return distance

    def compute_distance2(self, box_low, box_high, point):
        delta = 0.0#01
        temporary = (box_low + delta) - point
        term_1 = torch.max(temporary, torch.zeros(1).cuda() if temporary.is_cuda else torch.zeros(1))
        distance = torch.max(point - (box_high + delta), term_1)
        distance, _ = distance.max(dim=-1)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        z = (distance == 0).to(device, dtype=torch.float32)
        nz = (distance > 0).to(device, dtype=torch.float32)
        #print(point.shape,box_low.shape,((point*point).sum(dim=-1)).shape, ((point*box_low).sum(dim=-1)).shape)
        #print(z.shape, nz.shape)
        distance = (z*((point*point).sum(dim=-1))) + (nz*(torch.max((point*box_low).sum(dim=-1), (point*box_high).sum(dim=-1))))
        #distance, _ = distance.max(dim=-1)

        return distance

    def forward(self, s, r, o, flag_debug=0):
        base_forward = self.base_model(s, r, o)
        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)

        r_ht_low = self.R_ht_low(r)
        r_ht_high = self.R_ht_high(r)
        r_tt_low = self.R_tt_low(r)
        r_tt_high = self.R_tt_high(r)

        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)

        head_type_compatibility = - self.compute_distance2(r_ht_low, r_ht_high, s_t)#(s_t*r_ht).sum(-1)
        tail_type_compatibility = - self.compute_distance2(r_tt_low, r_tt_high, o_t)#(o_t*r_tt).sum(-1)

        base_forward = torch.nn.Sigmoid()(self.psi*base_forward)
        head_type_compatibility = torch.nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = torch.nn.Sigmoid()(self.psi*tail_type_compatibility)

        ###
        if flag_debug:
            score_data = base_forward
            print(score_data.shape)
            print_fun("Scores: Base: Mean: "+str(score_data.mean())+" Median: "+str(score_data.median()[0])+" STD: "+str(score_data.std()[0])+" MAX: "+str(torch.max(score_data))+" MIN: "+str(torch.min(score_data)))
            score_data=head_type_compatibility
            print_fun("Scores: Head: Mean: "+str(score_data.mean())+" Median: "+str(score_data.median()[0])+" STD: "+str(score_data.std()[0])+" MAX: "+str(torch.max(score_data))+" MIN: "+str(torch.min(score_data)))
            score_data=tail_type_compatibility
            print_fun("Scores: Tail: Mean: "+str(score_data.mean())+" Median: "+str(score_data.median()[0])+" STD: "+str(score_data.std()[0])+" MAX: "+str(torch.max(score_data))+" MIN: "+str(torch.min(score_data)))
        if flag_debug>1:
            #BOX SIZE
            box_sizes_tt = self.R_tt_high.weight.data - self.R_tt_low.weight.data
            box_sizes_ht = self.R_ht_high.weight.data - self.R_ht_low.weight.data

            box_size_max = torch.max(box_sizes_tt.abs(),1)[0]
            print_fun("Tail box size: Mean: "+str((box_size_max.mean(0)))+" Median: "+str((box_size_max.median(0))[0])+" STD: "+str((box_size_max.std(0))[0])+" Max: "+str(torch.max(box_size_max))+" Min: "+str(torch.min(box_size_max)))
            box_size_max = torch.max(box_sizes_ht.abs(),1)[0]
            print_fun("Head box size: Mean: "+str((box_size_max.mean(0)))+" Median: "+str((box_size_max.median(0))[0])+" STD: "+str((box_size_max.std(0))[0])+" Max: "+str(torch.max(box_size_max))+" Min: "+str(torch.min(box_size_max)))

        ###

        return self.mult*base_forward*head_type_compatibility*tail_type_compatibility

    def regularizer(self, s, r, o):
        box_sizes_tt = self.R_tt_high.weight.data - self.R_tt_low.weight.data
        box_sizes_ht = self.R_ht_high.weight.data - self.R_ht_low.weight.data
        if(self.box_reg == 'l1'):
            reg = (box_sizes_ht.abs() + box_sizes_tt.abs()).sum() #l1
        elif (self.box_reg == 'l2'):
            reg = (box_sizes_ht*box_sizes_ht + box_sizes_tt*box_sizes_tt).sum()
            #
            reg += (self.E_t.weight.data * self.E_t.weight.data).sum()
        else:
            utils.colored_print("red", "unknown regularizer" + str(self.reg))
        return reg * self.box_reg_coef + self.base_model.regularizer(s, r, o)

    def post_epoch(self):
        self.R_ht_high.weight.data, self.R_ht_low.weight.data = torch.max(self.R_ht_high.weight.data, self.R_ht_low.weight.data), torch.min(self.R_ht_high.weight.data, self.R_ht_low.weight.data)
        self.R_tt_high.weight.data, self.R_tt_low.weight.data = torch.max(self.R_tt_high.weight.data, self.R_tt_low.weight.data), torch.min(self.R_tt_high.weight.data, self.R_tt_low.weight.data)
        return self.base_model.post_epoch()






class box_typed_model2(torch.nn.Module):#box model implemented differently
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, mult=20.0, box_reg_coef=0.1, box_reg='l2', psi=2.0):
        super(box_typed_model2, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.mult = mult
        self.box_reg = box_reg
        self.box_reg_coef = box_reg_coef
        self.psi = psi
        self.E_t = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_ht_width = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_ht = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tt_width = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tt = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        torch.nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht_width.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt_width.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        #ensuring _low _high are init appropriately..low is low and vice versa
        #self.R_ht_width = torch.nn.functional.relu(self.R_ht_width)
        #self.R_tt_width = torch.nn.functional.relu(self.R_tt_width)
        self.minimum_value = 0.0

    def compute_distance(self, box, box_width, point):
        delta = 0.001
        temporary = (box + delta) - point
        term_1 = torch.max(temporary, torch.zeros(1).cuda() if temporary.is_cuda else torch.zeros(1))
        distance = torch.max(point - (box + box_width + delta), term_1)
        distance, _ = distance.max(dim=-1)
        return distance

    def compute_distance2(self, box_low, box_width, point):
        delta = 0.0#01
        box_high = box_low + box_width
        temporary = (box_low + delta) - point
        term_1 = torch.max(temporary, torch.zeros(1).cuda() if temporary.is_cuda else torch.zeros(1))
        distance = torch.max(point - (box_high + delta), term_1)
        distance, _ = distance.max(dim=-1)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        z = (distance == 0).to(device, dtype=torch.float32)
        nz = (distance > 0).to(device, dtype=torch.float32)
        #print(point.shape,box_low.shape,((point*point).sum(dim=-1)).shape, ((point*box_low).sum(dim=-1)).shape)
        #print(z.shape, nz.shape)
        distance = (z*((point*point).sum(dim=-1))) + (nz*(torch.max((point*box_low).sum(dim=-1), (point*box_high).sum(dim=-1))))
        #distance, _ = distance.max(dim=-1)

        return distance

    def forward(self, s, r, o, flag_debug=0):
        base_forward = self.base_model(s, r, o)
        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)

        r_ht = self.R_ht(r)
        r_ht_width = torch.nn.functional.relu(self.R_ht_width(r))#width should be positive
        r_tt = self.R_tt(r)
        r_tt_width = torch.nn.functional.relu(self.R_tt_width(r))

        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)

        head_type_compatibility = - self.compute_distance2(r_ht, r_ht_width, s_t)#(s_t*r_ht).sum(-1)
        tail_type_compatibility = - self.compute_distance2(r_tt, r_tt_width, o_t)#(o_t*r_tt).sum(-1)

        base_forward = torch.nn.Sigmoid()(self.psi*base_forward)
        head_type_compatibility = torch.nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = torch.nn.Sigmoid()(self.psi*tail_type_compatibility)

        ###
        if flag_debug:
            score_data = base_forward
            print(score_data.shape)
            print_fun("Scores: Base: Mean: "+str(score_data.mean())+" Median: "+str(score_data.median()[0])+" STD: "+str(score_data.std()[0])+" MAX: "+str(torch.max(score_data))+" MIN: "+str(torch.min(score_data)))
            score_data=head_type_compatibility
            print_fun("Scores: Head: Mean: "+str(score_data.mean())+" Median: "+str(score_data.median()[0])+" STD: "+str(score_data.std()[0])+" MAX: "+str(torch.max(score_data))+" MIN: "+str(torch.min(score_data)))
            score_data=tail_type_compatibility
            print_fun("Scores: Tail: Mean: "+str(score_data.mean())+" Median: "+str(score_data.median()[0])+" STD: "+str(score_data.std()[0])+" MAX: "+str(torch.max(score_data))+" MIN: "+str(torch.min(score_data)))
        if flag_debug>1:
            #BOX SIZE
            box_sizes_tt = self.R_tt_width.weight.data#self.R_tt_high.weight.data - self.R_tt_low.weight.data
            box_sizes_ht = self.R_ht_width.weight.data#self.R_ht_high.weight.data - self.R_ht_low.weight.data

            box_size_max = torch.max(box_sizes_tt.abs(),1)[0]
            print_fun("Tail box size: Mean: "+str((box_size_max.mean(0)))+" Median: "+str((box_size_max.median(0))[0])+" STD: "+str((box_size_max.std(0))[0])+" Max: "+str(torch.max(box_size_max))+" Min: "+str(torch.min(box_size_max)))
            box_size_max = torch.max(box_sizes_ht.abs(),1)[0]
            print_fun("Head box size: Mean: "+str((box_size_max.mean(0)))+" Median: "+str((box_size_max.median(0))[0])+" STD: "+str((box_size_max.std(0))[0])+" Max: "+str(torch.max(box_size_max))+" Min: "+str(torch.min(box_size_max)))

        ###

        return self.mult*base_forward*head_type_compatibility*tail_type_compatibility

    def regularizer(self, s, r, o):
        box_sizes_tt = self.R_tt_width.weight.data#self.R_tt_high.weight.data - self.R_tt_low.weight.data
        box_sizes_ht = self.R_ht_width.weight.data#self.R_ht_high.weight.data - self.R_ht_low.weight.data
        if(self.box_reg == 'l1'):
            reg = (box_sizes_ht.abs() + box_sizes_tt.abs()).sum() #l1
        elif (self.box_reg == 'l2'):
            reg = (box_sizes_ht*box_sizes_ht + box_sizes_tt*box_sizes_tt).sum()
            #
            #reg += (self.E_t.weight.data * self.E_t.weight.data).sum()
        else:
            utils.colored_print("red", "unknown regularizer" + str(self.reg))
        return reg * self.box_reg_coef + self.base_model.regularizer(s, r, o)

    def post_epoch(self):
        #self.R_ht_high.weight.data, self.R_ht_low.weight.data = torch.max(self.R_ht_high.weight.data, self.R_ht_low.weight.data), torch.min(self.R_ht_high.weight.data, self.R_ht_low.weight.data)
        #self.R_tt_high.weight.data, self.R_tt_low.weight.data = torch.max(self.R_tt_high.weight.data, self.R_tt_low.weight.data), torch.min(self.R_tt_high.weight.data, self.R_tt_low.weight.data)
        return ''#self.base_model.post_epoch()
