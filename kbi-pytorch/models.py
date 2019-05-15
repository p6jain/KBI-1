import torch
import utils
import random
import torchvision.models as torchvision_models
import torch.nn as nn
import torch.nn.functional as F
import numpy 

class distmult(torch.nn.Module):
    """
    DistMult Model from Trullion et al 2014.\n
    Scoring function (s, r, o) = <s, r, o> # dot product
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False, reg=2, flag_add_reverse=0):
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
    def __init__(self, entity_count, relation_count, embedding_dim, clamp_v=None, flag_add_reverse=0, with_sigmoid=0, unit_reg=0):
        super(complex, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count

        self.unit_reg = unit_reg

        self.E_im = torch.nn.Embedding(self.entity_count, self.embedding_dim, sparse=True)
        self.R_im = torch.nn.Embedding(self.relation_count, self.embedding_dim, sparse=True)
        self.E_re = torch.nn.Embedding(self.entity_count, self.embedding_dim, sparse=True)
        self.R_re = torch.nn.Embedding(self.relation_count, self.embedding_dim, sparse=True)
        '''
        torch.nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        '''
        self.E_im.weight.data *= 1e-3
        self.R_im.weight.data *= 1e-3
        self.E_re.weight.data *= 1e-3
        self.R_re.weight.data *= 1e-3
        #'''
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v
        
        self.flag_with_sigmoid =  with_sigmoid
        if flag_add_reverse:
            print("Complex:","Handling inverse relations as well")
        if with_sigmoid:
            print("Complex:","score with sigmoid")

    def forward(self, s, r, o, flag_debug=0):
        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)
        '''
        if self.clamp_v:
            s_im.data.clamp_(-self.clamp_v, self.clamp_v)
            s_re.data.clamp_(-self.clamp_v, self.clamp_v)
            r_im.data.clamp_(-self.clamp_v, self.clamp_v)
            r_re.data.clamp_(-self.clamp_v, self.clamp_v)
            o_im.data.clamp_(-self.clamp_v, self.clamp_v)
            o_re.data.clamp_(-self.clamp_v, self.clamp_v)
        '''
        if o is None:
            #x = s_im*r_re
            #y = (s_im*r_re+s_re*r_im) * o_im
            #print("Part 1")
            #print("Prachi Debug", "s_im", s_im.shape, "s_im*r_re", x.shape, "o_im", o_im.shape, "(s_im*r_re+s_re*r_im) * o_im", y.shape, y.sum(dim=-1).shape)
            tmp1 = (s_im*r_re+s_re*r_im); tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = (s_re*r_re-s_im*r_im); tmp2 = tmp2.view(-1,self.embedding_dim)
            o_re = o_re.view(-1,self.embedding_dim).transpose(0,1)
            o_im = o_im.view(-1,self.embedding_dim).transpose(0,1)
            result = tmp1 @ o_im + tmp2 @o_re #(s_im*r_re+s_re*r_im) @ o_im + (s_re*r_re-s_im*r_im) @ o_re
            #print("Prachi Debug", "result", result.shape)
            #print("End\n\n")
            #return result
        else:
            #result = s_im*(o_im*r_re-o_re*r_im) + s_re*(o_im*r_im+o_re*r_re) 
            #x = o_im*r_re
            #y = (o_im*r_re-o_re*r_im) * s_im
            #print("Part 2")
            #print("Prachi Debug", "o_im", o_im.shape, "o_im*r_re", x.shape, "s_im", s_im.shape, "(o_im*r_re-o_re*r_im) * s_im", y.shape, y.sum(dim=-1).shape)

            tmp1 = o_im*r_re-o_re*r_im; tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = o_im*r_im+o_re*r_re; tmp2 = tmp2.view(-1,self.embedding_dim)
            s_im = s_im.view(-1,self.embedding_dim).transpose(0,1)
            s_re = s_re.view(-1,self.embedding_dim).transpose(0,1) 
            result = tmp1 @ s_im + tmp2 @ s_re
    
            #print("Prachi Debug", "result", result.shape)
            #print("End\n\n")
        if flag_debug:
            print("@Prachi Debug", "result",result[0])
            print("@Prachi Debug", "result, mean, std",torch.mean(result),torch.std(result))

        if self.flag_with_sigmoid:
            result = torch.nn.Sigmoid()(result)

            if flag_debug:
                print("@Prachi Debug", "result",result[0])
                print("@Prachi Debug", "result, mean, std",torch.mean(result),torch.std(result))

        return result
        '''
        #result = (s_re*o_re+s_im*o_im)*r_re + (s_re*o_im-s_im*o_re)*r_im
        #result = o_im*(s_im*r_re+s_re*r_im) + o_re*(s_re*r_re-s_im*r_im)
            #return result.sum(dim=-1)



        r_im = self.R_im(r)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_re = self.R_re(r)
        o_re = self.E_re(o) if o is not None else self.E_im.weight.unsqueeze(0)
        #print("s.shape,o.shape,r.shape ",s.shape,o.shape,r.shape,s.shape[-1],o.shape[-1],r.shape[-1])
        #return (s_re*s_re+o_re*o_re+r_re*r_re+s_im*s_im+r_im*r_im+o_im*o_im).sum()
        return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2).sum()
        #denom_s = r.shape[0] * self.embedding_dim * s.shape[-1]
        #denom_o = r.shape[0] * self.embedding_dim * o.shape[-1]
        #return ((s_re*s_re).sum()+(s_im*s_im).sum())/(2.0*denom_s)+((o_im*o_im).sum()+(o_re*o_re).sum())/(2.0*denom_o)+(r_re*r_re).mean()/3.0+(r_im*r_im).mean()/3.0
        #return ((s_re*s_re).mean()+(s_im*s_im).mean()+(o_im*o_im).mean()+(o_re*o_re).mean())/(2.0)+((r_re*r_re).mean()+(r_im*r_im).mean())/4.0
        '''
    def regularizer_icml(self, s, r, o):#, s_wt, r_wt, o_wt):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        '''
        print("Prachi Debug","s",s_im.shape,s_wt.shape)
        print("Prachi Debug","r",r_im.shape,r_wt.shape)
        print("Prachi Debug","o",o_im.shape,o_wt.shape)
        '''
        #print("s.shape,o.shape,r.shape ",s.shape,o.shape,r.shape,s.shape[-1],o.shape[-1],r.shape[-1])
        factor = [torch.sqrt(s_re**2 + s_im**2),torch.sqrt(o_re**2+o_im**2),torch.sqrt(r_re**2+r_im**2)]
        reg = 0
        for ele in factor:
            reg += torch.sum(torch.abs(ele) ** 3)

        #print("Prachi Debug","reg",reg.shape, reg, s.shape, reg/s.shape[0])
        #print("Prachi Debug","ele",ele.shape)
        return reg/s.shape[0]
        #(s_wt*s_re*s_re+o_wt*o_re*o_re+r_wt*r_re*r_re+s_wt*s_im*s_im+r_wt*r_im*r_im+o_wt*o_im*o_im).sum()

    def regularizer(self, s, r, o):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        return (s_re*s_re+o_re*o_re+r_re*r_re+s_im*s_im+r_im*r_im+o_im*o_im).sum()

    def regularizer_icml_orig(self, s, r, o, s_wt, r_wt, o_wt):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)

        s_r = s_wt*torch.sqrt(s_re*s_re + s_im*s_im)
        r_r = r_wt*torch.sqrt(r_re*r_re + r_im*r_im)
        o_r = o_wt*torch.sqrt(o_re*o_re + o_im*o_im)
        return (s_r+r_r+o_r).sum()


    def post_epoch(self):
        if(self.unit_reg):
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
        return ""


class adder_model(torch.nn.Module):
    def __init__(self, entity_count, relation_count, model1_name, model1_arguments, model2_name, model2_arguments, flag_add_reverse=0):
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

class model_reflexive(torch.nn.Module):
    def __init__(self, entity_count, relation_count, base_model_name, base_model_arguments, flag_add_reverse=0):
        '''
        \beta (right reweighing of type and base model)  and epsilon (handle reflexivity) model
        '''

        super(model_reflexive, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = self.base_model.embedding_dim

        self.entity_count = entity_count
        if 0:#flag_add_reverse:
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count

        self.minimum_value = 0.0
        
        self.flag_add_reverse=flag_add_reverse

        #reflexive
        self.eps = torch.nn.Embedding(self.relation_count, 1)
        torch.nn.init.constant_(self.eps.weight.data, -3.0)
        ##
        

    def forward(self, s, r, o, flag_debug=0, beta_tmp = None):
        base_forward = self.base_model(s, r, o)

        if 0:#flag_add_reverse:
            total_rel = torch.tensor(self.relation_count).cuda() 
            inv_or_not = r >= total_rel; #inv_or_not = inv_or_not.type(torch.LongTensor)
            r = r - inv_or_not.type(torch.cuda.LongTensor) * total_rel

        #base_forward = torch.nn.Sigmoid()(self.psi*base_forward)
        epsilon = self.eps(r).squeeze(2)
        eps = torch.nn.Sigmoid()(epsilon)

        score_old = base_forward 

        if s is None:
            base_o = score_old.gather(1, o)
            score_new = score_old.scatter_(1, o, base_o*eps)
            return score_new
        if o is None:
            base_s = score_old.gather(1, s)
            score_new = score_old.scatter_(1, s, base_s*eps)
            return score_new

        #score_new = eps * ((score_old * (s==o).type(torch.cuda.FloatTensor)) + (score_old * (s != o).type(torch.cuda.FloatTensor))) + ((1 - eps) * (score_old * (s != o).type(torch.cuda.FloatTensor)) )
        score_new = eps * (score_old * (s==o).type(torch.cuda.FloatTensor)) + (score_old * (s != o).type(torch.cuda.FloatTensor)) 

        return score_new


    def regularizer(self, s, r, o):
        return self.base_model.regularizer(s, r, o)


    def regularizer_icml(self, s, r, o):#, s_wt, r_wt, o_wt):
        return self.base_model.regularizer_icml(s, r, o)

class typed_model_v2(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, flag_add_reverse=0, flag_train_beta=0, best_beta=None, base_reg=None, type_reg=None):
        '''
        \beta (right reweighing of type and base model)  and epsilon (handle reflexivity) model
        '''

        super(typed_model_v2, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        if flag_add_reverse:
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.mult = mult
        self.psi = psi
        self.E_t = torch.nn.Embedding(self.entity_count, self.embedding_dim, sparse=True)
        self.R_ht = torch.nn.Embedding(self.relation_count, self.embedding_dim, sparse=True)
        self.R_tt = torch.nn.Embedding(self.relation_count, self.embedding_dim, sparse=True)

        #'''
        torch.nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        '''

        self.E_t.weight.data *= 1e-3
        self.R_ht.weight.data *= 1e-3
        self.R_tt.weight.data *= 1e-3
        '''
        self.base_reg = base_reg
        self.type_reg = type_reg
        self.minimum_value = 0.0
        
        self.flag_add_reverse=flag_add_reverse

        ##
        self.flag_train_beta = flag_train_beta
        #better combination - convex
        self.beta = torch.nn.Embedding(self.relation_count, 1)
        if best_beta is not None:
            print("Using Best Beta!")
            self.beta.weight.data.copy_(torch.from_numpy(best_beta).unsqueeze(1)) 
        else:
            torch.nn.init.constant_(self.beta.weight.data, 3.0)
        print("Prachi Debug", "best_beta", best_beta) 
        #reflexive
        self.eps = torch.nn.Embedding(self.relation_count, 1)
        torch.nn.init.constant_(self.eps.weight.data, -3.0)
        ##
        if flag_add_reverse:
            print("Typed model v2:","Handling inverse relations as well") 

    def forward(self, s, r, o, flag_debug=0, beta_tmp = None):
        base_forward = self.base_model(s, r, o)


        if self.flag_add_reverse:
            total_rel = torch.tensor(self.relation_count).cuda()
            inv_or_not = r >= total_rel; #inv_or_not = inv_or_not.type(torch.LongTensor)
            r = r - inv_or_not.type(torch.cuda.LongTensor) * total_rel


        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)

        r_tt = r_tt.view(-1,self.embedding_dim)
        r_ht = r_ht.view(-1,self.embedding_dim)

        r_ht_new = torch.where(inv_or_not, r_tt, r_ht)
        r_tt_new = torch.where(inv_or_not, r_ht, r_tt)
        r_tt = r_tt_new; r_ht = r_ht_new; r_ht_new= None; r_tt_new=None

        r_ht = r_ht.unsqueeze(1)
        r_tt = r_tt.unsqueeze(1)

        if s is None:
            s_t = s_t.view(-1,self.embedding_dim)
            r_ht = r_ht.view(-1,self.embedding_dim)
            head_type_compatibility = r_ht @ s_t.transpose(0,1) 
        else:
            head_type_compatibility = (s_t*r_ht).sum(-1)
        if o is None:
            o_t = o_t.view(-1,self.embedding_dim)
            r_tt = r_tt.view(-1,self.embedding_dim)
            tail_type_compatibility = r_tt @ o_t.transpose(0,1)
        else:
            tail_type_compatibility = (o_t*r_tt).sum(-1)

        #base_forward = torch.nn.Sigmoid()(self.psi*base_forward)
        head_type_compatibility = torch.nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = torch.nn.Sigmoid()(self.psi*tail_type_compatibility)

        #return self.mult*base_forward*head_type_compatibility*tail_type_compatibility #, base_forward, head_type_compatibility, tail_type_compatibility
        ##
        epsilon = self.eps(r).squeeze(2)
        eps = torch.nn.Sigmoid()(epsilon)


        if beta_tmp is None:
            if self.flag_train_beta==0:
                beta=1
            else:
                betas = self.beta(r).squeeze(2)
                beta = torch.nn.Sigmoid()(betas)
        else:
            beta = beta_tmp

        score_old = (base_forward*beta + 1.0 - beta)*head_type_compatibility*tail_type_compatibility

        if s is None:
            base_o = score_old.gather(1, o)
            score_new = score_old.scatter_(1, o, base_o*eps)
            return self.mult*score_new
        if o is None:
            base_s = score_old.gather(1, s)
            score_new = score_old.scatter_(1, s, base_s*eps)
            return self.mult*score_new

        #score_new = eps * ((score_old * (s==o).type(torch.cuda.FloatTensor)) + (score_old * (s != o).type(torch.cuda.FloatTensor))) + ((1 - eps) * (score_old * (s != o).type(torch.cuda.FloatTensor)) )
        score_new = eps * (score_old * (s==o).type(torch.cuda.FloatTensor)) + (score_old * (s != o).type(torch.cuda.FloatTensor)) 

        return self.mult*score_new
        ##


    def regularizer(self, s, r, o):
        ''' 
        s_t = self.E_t(s)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o)
        reg = (s_t*s_t + r_ht*r_ht + r_tt*r_tt + o_t*o_t).sum()
        return self.base_model.regularizer(s, r, o) + reg
        '''
        if self.flag_train_beta:
            beta = torch.nn.Sigmoid()(self.beta(r))
            #print("Prachi Debug -- reg, beta shape",self.base_model.regularizer(s,r,o).shape, beta.shape)
            return self.base_model.regularizer(s, r, o)#+ (beta**2).sum()
        else:
            return self.base_model.regularizer(s, r, o)


    def regularizer_icml(self, s, r, o):#, s_wt, r_wt, o_wt):
        s_t = self.E_t(s)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o)
        #reg = (s_t*s_t + r_ht*r_ht + r_tt*r_tt + o_t*o_t).sum()
        #return self.base_model.regularizer(s, r, o) + reg
        
        factor = [torch.sqrt(s_t**2),torch.sqrt(o_t**2),torch.sqrt(r_ht**2+r_tt**2)]
        reg = 0
        for ele in factor:
            reg += torch.sum(torch.abs(ele) ** 3)

        #print("Prachi Debug","reg",reg.shape, reg, s.shape, reg/s.shape[0])
        #print("Prachi Debug","ele",ele.shape)
        #return reg/s.shape[0] + self.base_model.regularizer_icml(s, r, o)
        return self.type_reg*(reg/s.shape[0]) + self.base_reg*(self.base_model.regularizer_icml(s, r, o))
        #return self.base_model.regularizer_icml(s, r, o)

    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
        return self.base_model.post_epoch()



class typed_model_v1_DAG(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, flag_add_reverse=0, flag_train_beta=1, best_beta=None, base_reg=1.0, type_reg=1.0, base_model_bias=0.0):
        '''
        #to do alpha gamma here
        #\beta (right reweighing of type and base model) + normalize by dim : D + for scale shift alpha gamma 
        '''

        super(typed_model_v1_DAG, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model_dim = base_model_arguments["embedding_dim"]
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        if flag_add_reverse:
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.mult = mult
        self.psi = psi
        self.E_t = torch.nn.Embedding(self.entity_count, self.embedding_dim, sparse=True)
        self.R_ht = torch.nn.Embedding(self.relation_count, self.embedding_dim, sparse=True)
        self.R_tt = torch.nn.Embedding(self.relation_count, self.embedding_dim, sparse=True)

        #'''
        torch.nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        '''

        self.E_t.weight.data *= 1e-3
        self.R_ht.weight.data *= 1e-3
        self.R_tt.weight.data *= 1e-3
        '''
        self.base_reg = base_reg
        self.type_reg = type_reg
        self.minimum_value = 0.0
        
        self.flag_add_reverse=flag_add_reverse

        ##
        self.flag_train_beta = flag_train_beta
        if flag_train_beta:
            print("Training beta values")

        self.base_model_bias = base_model_bias
        if base_model_bias: 
            print("Base model bias", base_model_bias)

        #better combination - convex
        self.beta = torch.nn.Embedding(self.relation_count, 1)
        if best_beta is not None:
            print("Using Best Beta!")
            self.beta.weight.data.copy_(torch.from_numpy(best_beta).unsqueeze(1)) 
        else:
            torch.nn.init.constant_(self.beta.weight.data, 3.0)
        print("Prachi Debug", "best_beta", best_beta) 
        

        self.w_base = torch.nn.Embedding(1, 1)
        self.b_base = torch.nn.Embedding(1, 1)
        torch.nn.init.constant_(self.b_base.weight.data, 0.0)
        torch.nn.init.constant_(self.w_base.weight.data, 0.25)
        self.w_head = torch.nn.Embedding(1, 1)
        self.b_head = torch.nn.Embedding(1, 1)
        torch.nn.init.constant_(self.b_head.weight.data, 0.0)
        torch.nn.init.constant_(self.w_head.weight.data, 0.25)
        self.w_tail = torch.nn.Embedding(1, 1)
        self.b_tail = torch.nn.Embedding(1, 1)
        torch.nn.init.constant_(self.b_tail.weight.data, 0.0)
        torch.nn.init.constant_(self.w_tail.weight.data, 0.25)

    def forward(self, s, r, o, flag_debug=0, beta_tmp = None):
        base_forward = self.base_model(s, r, o)

        if 1:#self.flag_add_reverse:
            total_rel = torch.tensor(self.relation_count).cuda() 
            inv_or_not = r >= total_rel; #inv_or_not = inv_or_not.type(torch.LongTensor)
            r = r - inv_or_not.type(torch.cuda.LongTensor) * total_rel

        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)

        r_tt = r_tt.view(-1,self.embedding_dim)
        r_ht = r_ht.view(-1,self.embedding_dim)

        r_ht_new = torch.where(inv_or_not, r_tt, r_ht)
        r_tt_new = torch.where(inv_or_not, r_ht, r_tt)
        r_tt = r_tt_new; r_ht = r_ht_new; r_ht_new= None; r_tt_new=None

        r_ht = r_ht.unsqueeze(1)
        r_tt = r_tt.unsqueeze(1)

        if s is None:
            s_t = s_t.view(-1,self.embedding_dim)
            r_ht = r_ht.view(-1,self.embedding_dim)
            head_type_compatibility = r_ht @ s_t.transpose(0,1) 
        else:
            head_type_compatibility = (s_t*r_ht).sum(-1)
        if o is None:
            o_t = o_t.view(-1,self.embedding_dim)
            r_tt = r_tt.view(-1,self.embedding_dim)
            tail_type_compatibility = r_tt @ o_t.transpose(0,1)
        else:
            tail_type_compatibility = (o_t*r_tt).sum(-1)

        if flag_debug:
            utils.colored_print("yellow", "\nBefore Sigmoid")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))
        base_forward = (self.psi*base_forward - self.b_base.weight) #* self.w_base.weight  
        #head_type_compatibility = self.w_head.weight * head_type_compatibility + self.b_head.weight
        #tail_type_compatibility = self.w_tail.weight * tail_type_compatibility + self.b_tail.weight
 

        if flag_debug:
            utils.colored_print("green", "Before Sigmoid: After scale shift")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward), self.w_base.weight, self.b_base.weight)
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility), self.w_head.weight, self.b_head.weight)
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility), self.w_tail.weight, self.b_tail.weight)

        self.psi = 1.0
        base_forward = torch.nn.Sigmoid()(self.psi*base_forward) #+ self.base_model_bias
        self.psi = 1.0
        head_type_compatibility = torch.nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = torch.nn.Sigmoid()(self.psi*tail_type_compatibility)

        if beta_tmp is None:
            if self.flag_train_beta==0:
                beta=1
            else:
                betas = self.beta(r).squeeze(2)
                beta = torch.nn.Sigmoid()(betas)
        else:
            beta = beta_tmp

        score_old = (base_forward*beta + 1.0 - beta)*head_type_compatibility*tail_type_compatibility

        if flag_debug:
            utils.colored_print("blue", "After Sigmoid: mean:" + str(torch.mean(score_old)) + " std:" + str(torch.std(score_old)))
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))

        self.w_base.weight.data.clamp_(-0.25, 0.25)
        self.b_base.weight.data.clamp_(-0.25, 0.25)
        self.w_tail.weight.data.clamp_(-0.25, 0.25)
        self.b_tail.weight.data.clamp_(-0.25, 0.25)
        self.w_head.weight.data.clamp_(-0.25, 0.25)
        self.b_head.weight.data.clamp_(-0.25, 0.25)


        return self.mult*score_old


    def regularizer(self, s, r, o):
        ''' 
        s_t = self.E_t(s)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o)
        reg = (s_t*s_t + r_ht*r_ht + r_tt*r_tt + o_t*o_t).sum()
        return self.base_model.regularizer(s, r, o) + reg
        '''
        if self.flag_train_beta:
            beta = torch.nn.Sigmoid()(self.beta(r))
            #print("Prachi Debug -- reg, beta shape",self.base_model.regularizer(s,r,o).shape, beta.shape)
            return self.base_model.regularizer(s, r, o)#+ (beta**2).sum()
        else:
            return self.base_model.regularizer(s, r, o)


    def regularizer_icml(self, s, r, o):#, s_wt, r_wt, o_wt):
        s_t = self.E_t(s)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o)
        #reg = (s_t*s_t + r_ht*r_ht + r_tt*r_tt + o_t*o_t).sum()
        #return self.base_model.regularizer(s, r, o) + reg
        
        factor = [torch.sqrt(s_t**2),torch.sqrt(o_t**2),torch.sqrt(r_ht**2+r_tt**2)]
        reg = 0
        for ele in factor:
            reg += torch.sum(torch.abs(ele) ** 3)

        #print("Prachi Debug","reg",reg.shape, reg, s.shape, reg/s.shape[0])
        #print("Prachi Debug","ele",ele.shape)
        #return reg/s.shape[0] + self.base_model.regularizer_icml(s, r, o)
        return self.type_reg*(reg/s.shape[0]) + self.base_reg*(self.base_model.regularizer_icml(s, r, o))
        #return self.base_model.regularizer_icml(s, r, o)

    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
        return self.base_model.post_epoch()



class typed_model_v1_ss(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, flag_add_reverse=0, flag_train_beta=0, best_beta=None, base_reg=1.0, type_reg=1.0, base_model_bias=0.0):
        '''
        \beta (right reweighing of type and base model) + and scale shift  
        '''

        super(typed_model_v1_ss, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        if flag_add_reverse:
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.mult = mult
        self.psi = psi
        self.E_t = torch.nn.Embedding(self.entity_count, self.embedding_dim, sparse=True)
        self.R_ht = torch.nn.Embedding(self.relation_count, self.embedding_dim, sparse=True)
        self.R_tt = torch.nn.Embedding(self.relation_count, self.embedding_dim, sparse=True)

        #'''
        torch.nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        '''

        self.E_t.weight.data *= 1e-3
        self.R_ht.weight.data *= 1e-3
        self.R_tt.weight.data *= 1e-3
        '''
        self.base_reg = base_reg
        self.type_reg = type_reg
        self.minimum_value = 0.0
        
        self.flag_add_reverse=flag_add_reverse

        ##
        self.flag_train_beta = flag_train_beta
        if flag_train_beta:
            print("Training beta values")

        self.base_model_bias = base_model_bias
        if base_model_bias: 
            print("Base model bias", base_model_bias)

        #better combination - convex
        self.beta = torch.nn.Embedding(self.relation_count, 1)
        if best_beta is not None:
            print("Using Best Beta!")
            self.beta.weight.data.copy_(torch.from_numpy(best_beta).unsqueeze(1)) 
        else:
            torch.nn.init.constant_(self.beta.weight.data, 3.0)
        print("Prachi Debug", "best_beta", best_beta) 
        

        self.w_base = torch.nn.Embedding(1, 1)
        self.b_base = torch.nn.Embedding(1, 1)
        torch.nn.init.constant_(self.b_base.weight.data, 0.0)
        torch.nn.init.constant_(self.w_base.weight.data, 0.25)
        self.w_head = torch.nn.Embedding(1, 1)
        self.b_head = torch.nn.Embedding(1, 1)
        torch.nn.init.constant_(self.b_head.weight.data, 0.0)
        torch.nn.init.constant_(self.w_head.weight.data, 0.25)
        self.w_tail = torch.nn.Embedding(1, 1)
        self.b_tail = torch.nn.Embedding(1, 1)
        torch.nn.init.constant_(self.b_tail.weight.data, 0.0)
        torch.nn.init.constant_(self.w_tail.weight.data, 0.25)

    def forward(self, s, r, o, flag_debug=0, beta_tmp = None):
        base_forward = self.base_model(s, r, o)

        if 1:#self.flag_add_reverse:
            total_rel = torch.tensor(self.relation_count).cuda() 
            inv_or_not = r >= total_rel; #inv_or_not = inv_or_not.type(torch.LongTensor)
            r = r - inv_or_not.type(torch.cuda.LongTensor) * total_rel

        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)

        r_tt = r_tt.view(-1,self.embedding_dim)
        r_ht = r_ht.view(-1,self.embedding_dim)

        r_ht_new = torch.where(inv_or_not, r_tt, r_ht)
        r_tt_new = torch.where(inv_or_not, r_ht, r_tt)
        r_tt = r_tt_new; r_ht = r_ht_new; r_ht_new= None; r_tt_new=None

        r_ht = r_ht.unsqueeze(1)
        r_tt = r_tt.unsqueeze(1)

        if s is None:
            s_t = s_t.view(-1,self.embedding_dim)
            r_ht = r_ht.view(-1,self.embedding_dim)
            head_type_compatibility = r_ht @ s_t.transpose(0,1) 
        else:
            head_type_compatibility = (s_t*r_ht).sum(-1)
        if o is None:
            o_t = o_t.view(-1,self.embedding_dim)
            r_tt = r_tt.view(-1,self.embedding_dim)
            tail_type_compatibility = r_tt @ o_t.transpose(0,1)
        else:
            tail_type_compatibility = (o_t*r_tt).sum(-1)

        if flag_debug:
            utils.colored_print("yellow", "\nBefore Sigmoid")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))
        base_forward = self.w_base.weight * base_forward + self.b_base.weight
        head_type_compatibility = self.w_head.weight * head_type_compatibility + self.b_head.weight
        tail_type_compatibility = self.w_tail.weight * tail_type_compatibility + self.b_tail.weight
 

        if flag_debug:
            utils.colored_print("green", "Before Sigmoid: After scale shift")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward), self.w_base.weight, self.b_base.weight)
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility), self.w_head.weight, self.b_head.weight)
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility), self.w_tail.weight, self.b_tail.weight)

        base_forward = torch.nn.Sigmoid()(self.psi*base_forward) #+ self.base_model_bias
        self.psi = 1.0
        head_type_compatibility = torch.nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = torch.nn.Sigmoid()(self.psi*tail_type_compatibility)

        if beta_tmp is None:
            if self.flag_train_beta==0:
                beta=1
            else:
                betas = self.beta(r).squeeze(2)
                beta = torch.nn.Sigmoid()(betas)
        else:
            beta = beta_tmp

        score_old = (base_forward*beta + 1.0 - beta)*head_type_compatibility*tail_type_compatibility

        if flag_debug:
            utils.colored_print("blue", "After Sigmoid: mean:" + str(torch.mean(score_old)) + " std:" + str(torch.std(score_old)))
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))

        self.w_base.weight.data.clamp_(-0.25, 0.25)
        self.b_base.weight.data.clamp_(-0.25, 0.25)
        self.w_tail.weight.data.clamp_(-0.25, 0.25)
        self.b_tail.weight.data.clamp_(-0.25, 0.25)
        self.w_head.weight.data.clamp_(-0.25, 0.25)
        self.b_head.weight.data.clamp_(-0.25, 0.25)


        return self.mult*score_old


    def regularizer(self, s, r, o):
        ''' 
        s_t = self.E_t(s)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o)
        reg = (s_t*s_t + r_ht*r_ht + r_tt*r_tt + o_t*o_t).sum()
        return self.base_model.regularizer(s, r, o) + reg
        '''
        if self.flag_train_beta:
            beta = torch.nn.Sigmoid()(self.beta(r))
            #print("Prachi Debug -- reg, beta shape",self.base_model.regularizer(s,r,o).shape, beta.shape)
            return self.base_model.regularizer(s, r, o)#+ (beta**2).sum()
        else:
            return self.base_model.regularizer(s, r, o)


    def regularizer_icml(self, s, r, o):#, s_wt, r_wt, o_wt):
        s_t = self.E_t(s)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o)
        #reg = (s_t*s_t + r_ht*r_ht + r_tt*r_tt + o_t*o_t).sum()
        #return self.base_model.regularizer(s, r, o) + reg
        
        factor = [torch.sqrt(s_t**2),torch.sqrt(o_t**2),torch.sqrt(r_ht**2+r_tt**2)]
        reg = 0
        for ele in factor:
            reg += torch.sum(torch.abs(ele) ** 3)

        #print("Prachi Debug","reg",reg.shape, reg, s.shape, reg/s.shape[0])
        #print("Prachi Debug","ele",ele.shape)
        #return reg/s.shape[0] + self.base_model.regularizer_icml(s, r, o)
        return self.type_reg*(reg/s.shape[0]) + self.base_reg*(self.base_model.regularizer_icml(s, r, o))
        #return self.base_model.regularizer_icml(s, r, o)

    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
        return self.base_model.post_epoch()


class typed_model_v1(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, flag_add_reverse=0, flag_train_beta=1, best_beta=None, base_reg=1.0, type_reg=1.0, base_model_bias=0.0):
        '''
        \beta (right reweighing of type and base model)  
        '''

        super(typed_model_v1, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        if flag_add_reverse:
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.mult = mult
        self.psi = psi
        self.E_t = torch.nn.Embedding(self.entity_count, self.embedding_dim, sparse=True)
        self.R_ht = torch.nn.Embedding(self.relation_count, self.embedding_dim, sparse=True)
        self.R_tt = torch.nn.Embedding(self.relation_count, self.embedding_dim, sparse=True)

        #'''
        torch.nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        '''

        self.E_t.weight.data *= 1e-3
        self.R_ht.weight.data *= 1e-3
        self.R_tt.weight.data *= 1e-3
        '''
        self.base_reg = base_reg
        self.type_reg = type_reg
        self.minimum_value = 0.0
        
        self.flag_add_reverse=flag_add_reverse

        ##
        self.flag_train_beta = flag_train_beta
        if flag_train_beta:
            print("Training beta values")

        self.base_model_bias = base_model_bias
        if base_model_bias: 
            print("Base model bias", base_model_bias)

        #better combination - convex
        self.beta = torch.nn.Embedding(self.relation_count, 1)
        if best_beta is not None:
            print("Using Best Beta!")
            self.beta.weight.data.copy_(torch.from_numpy(best_beta).unsqueeze(1)) 
        else:
            torch.nn.init.constant_(self.beta.weight.data, 3.0)
        print("Prachi Debug", "best_beta", best_beta) 
        

    def forward(self, s, r, o, flag_debug=0, beta_tmp = None):
        base_forward = self.base_model(s, r, o)

        if 1:#self.flag_add_reverse:
            total_rel = torch.tensor(self.relation_count).cuda() 
            inv_or_not = r >= total_rel; #inv_or_not = inv_or_not.type(torch.LongTensor)
            r = r - inv_or_not.type(torch.cuda.LongTensor) * total_rel

        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)

        r_tt = r_tt.view(-1,self.embedding_dim)
        r_ht = r_ht.view(-1,self.embedding_dim)

        r_ht_new = torch.where(inv_or_not, r_tt, r_ht)
        r_tt_new = torch.where(inv_or_not, r_ht, r_tt)
        r_tt = r_tt_new; r_ht = r_ht_new; r_ht_new= None; r_tt_new=None

        r_ht = r_ht.unsqueeze(1)
        r_tt = r_tt.unsqueeze(1)

        if s is None:
            s_t = s_t.view(-1,self.embedding_dim)
            r_ht = r_ht.view(-1,self.embedding_dim)
            head_type_compatibility = r_ht @ s_t.transpose(0,1) 
        else:
            head_type_compatibility = (s_t*r_ht).sum(-1)
        if o is None:
            o_t = o_t.view(-1,self.embedding_dim)
            r_tt = r_tt.view(-1,self.embedding_dim)
            tail_type_compatibility = r_tt @ o_t.transpose(0,1)
        else:
            tail_type_compatibility = (o_t*r_tt).sum(-1)

        if flag_debug:
            print("Before Sigmoid")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))

        base_forward = torch.nn.Sigmoid()(self.psi*base_forward) + self.base_model_bias
        head_type_compatibility = torch.nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = torch.nn.Sigmoid()(self.psi*tail_type_compatibility)

        if beta_tmp is None:
            if self.flag_train_beta==0:
                beta=1
            else:
                betas = self.beta(r).squeeze(2)
                beta = torch.nn.Sigmoid()(betas)
        else:
            beta = beta_tmp

        score_old = (base_forward*beta + 1.0 - beta)*head_type_compatibility*tail_type_compatibility

        if flag_debug:
            print("After Sigmoid", torch.mean(score_old), torch.std(score_old))
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))


        return self.mult*score_old


    def regularizer(self, s, r, o):
        ''' 
        s_t = self.E_t(s)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o)
        reg = (s_t*s_t + r_ht*r_ht + r_tt*r_tt + o_t*o_t).sum()
        return self.base_model.regularizer(s, r, o) + reg
        '''
        if self.flag_train_beta:
            beta = torch.nn.Sigmoid()(self.beta(r))
            #print("Prachi Debug -- reg, beta shape",self.base_model.regularizer(s,r,o).shape, beta.shape)
            return self.base_model.regularizer(s, r, o)#+ (beta**2).sum()
        else:
            return self.base_model.regularizer(s, r, o)


    def regularizer_icml(self, s, r, o):#, s_wt, r_wt, o_wt):
        s_t = self.E_t(s)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o)
        #reg = (s_t*s_t + r_ht*r_ht + r_tt*r_tt + o_t*o_t).sum()
        #return self.base_model.regularizer(s, r, o) + reg
        
        factor = [torch.sqrt(s_t**2),torch.sqrt(o_t**2),torch.sqrt(r_ht**2+r_tt**2)]
        reg = 0
        for ele in factor:
            reg += torch.sum(torch.abs(ele) ** 3)

        #print("Prachi Debug","reg",reg.shape, reg, s.shape, reg/s.shape[0])
        #print("Prachi Debug","ele",ele.shape)
        #return reg/s.shape[0] + self.base_model.regularizer_icml(s, r, o)
        return self.type_reg*(reg/s.shape[0]) + self.base_reg*(self.base_model.regularizer_icml(s, r, o))
        #return self.base_model.regularizer_icml(s, r, o)

    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
        return self.base_model.post_epoch()

class typed_model_v3(torch.nn.Module):
    '''
    typed scores different
    '''
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, flag_add_reverse=0):
        super(typed_model_v3, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        if flag_add_reverse:
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.mult = mult
        self.psi = psi
        self.E_t = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_ht = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tt = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        '''
        torch.nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        '''
        self.E_t.weight.data *= 1e-3
        self.R_ht.weight.data *= 1e-3
        self.R_tt.weight.data *= 1e-3

        self.minimum_value = 0.0
        
        self.flag_add_reverse=flag_add_reverse

    def forward(self, s, r, o, flag_debug=0):
        base_forward = self.base_model(s, r, o)

        if self.flag_add_reverse:
            total_rel = torch.tensor(self.relation_count).cuda() 
            inv_or_not = r >= total_rel; #inv_or_not = inv_or_not.type(torch.LongTensor)
            r = r - inv_or_not.type(torch.cuda.LongTensor) * total_rel

        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)


        r_tt = r_tt.view(-1,self.embedding_dim)
        r_ht = r_ht.view(-1,self.embedding_dim)

        r_ht_new = torch.where(inv_or_not, r_tt, r_ht)
        r_tt_new = torch.where(inv_or_not, r_ht, r_tt)
        r_tt = r_tt_new; r_ht = r_ht_new; r_ht_new= None; r_tt_new=None

        r_ht = r_ht.unsqueeze(1)
        r_tt = r_tt.unsqueeze(1)

        if s is None:
            s_t = s_t.view(-1,self.embedding_dim)
            r_ht = r_ht.view(-1,self.embedding_dim)
            o_t = o_t.view(-1,self.embedding_dim)
            head_type_compatibility = (o_t * r_ht) @ s_t.transpose(0,1) 
        else:
            s_t = s_t.view(-1,self.embedding_dim)
            r_ht = r_ht.view(-1,self.embedding_dim)
            o_t = o_t.view(-1,self.embedding_dim)
            head_type_compatibility = (s_t*r_ht) @ o_t.transpose(0,1)
            #head_type_compatibility = (s_t*r_ht).sum(-1)

        if o is None:
            o_t = o_t.view(-1,self.embedding_dim)
            r_tt = r_tt.view(-1,self.embedding_dim)
            s_t = s_t.view(-1,self.embedding_dim)
            tail_type_compatibility = (s_t*r_tt) @ o_t.transpose(0,1)
        else:
            o_t = o_t.view(-1,self.embedding_dim)
            r_tt = r_tt.view(-1,self.embedding_dim)
            s_t = s_t.view(-1,self.embedding_dim)
            tail_type_compatibility = (o_t*r_tt) @ s_t.transpose(0,1)
            #tail_type_compatibility = (o_t*r_tt).sum(-1)

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

    def regularizer_icml(self, s, r, o):#, s_wt, r_wt, o_wt):
        #'''
        s_t = self.E_t(s)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o)
        #reg = (s_t*s_t + r_ht*r_ht + r_tt*r_tt + o_t*o_t).sum()
        #return self.base_model.regularizer(s, r, o) + reg
        
        factor = [torch.sqrt(s_t**2),torch.sqrt(o_t**2),torch.sqrt(r_ht**2+r_tt**2)]
        reg = 0
        for ele in factor:
            reg += torch.sum(torch.abs(ele) ** 3)

        #print("Prachi Debug","reg",reg.shape, reg, s.shape, reg/s.shape[0])
        #print("Prachi Debug","ele",ele.shape)
        return reg/s.shape[0] + self.base_model.regularizer_icml(s, r, o)
        #'''
        #return self.base_model.regularizer_icml(s, r, o)


    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
        return self.base_model.post_epoch()


class typed_model(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, flag_add_reverse=0, type_reg=1.0, base_reg=1.0, base_model_bias=0.0):
        super(typed_model, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        if flag_add_reverse:
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.mult = mult
        self.psi = psi
        self.E_t = torch.nn.Embedding(self.entity_count, self.embedding_dim, sparse=True)
        self.R_ht = torch.nn.Embedding(self.relation_count, self.embedding_dim, sparse=True)
        self.R_tt = torch.nn.Embedding(self.relation_count, self.embedding_dim, sparse=True)
        #'''
        torch.nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        '''
        self.E_t.weight.data *= 1e-3
        self.R_ht.weight.data *= 1e-3
        self.R_tt.weight.data *= 1e-3
        '''  
        self.minimum_value = 0.0
        
        self.flag_add_reverse=flag_add_reverse

        self.type_reg = type_reg
        self.base_reg = base_reg

        self.base_model_bias = base_model_bias
        if base_model_bias:
            print("base_model_bias", base_model_bias)

    def forward(self, s, r, o, flag_debug=0):
        base_forward = self.base_model(s, r, o)

        total_rel = torch.tensor(self.relation_count).cuda() 
        inv_or_not = r >= total_rel; #inv_or_not = inv_or_not.type(torch.LongTensor)
        r = r - inv_or_not.type(torch.cuda.LongTensor) * total_rel

        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)


        r_tt = r_tt.view(-1,self.embedding_dim)
        r_ht = r_ht.view(-1,self.embedding_dim)

        r_ht_new = torch.where(inv_or_not, r_tt, r_ht)
        r_tt_new = torch.where(inv_or_not, r_ht, r_tt)
        r_tt = r_tt_new; r_ht = r_ht_new; r_ht_new= None; r_tt_new=None

        r_ht = r_ht.unsqueeze(1)
        r_tt = r_tt.unsqueeze(1)

        if s is None:
            s_t = s_t.view(-1,self.embedding_dim)
            r_ht = r_ht.view(-1,self.embedding_dim)
            head_type_compatibility = r_ht @ s_t.transpose(0,1) 
        else:
            head_type_compatibility = (s_t*r_ht).sum(-1)
        if o is None:
            o_t = o_t.view(-1,self.embedding_dim)
            r_tt = r_tt.view(-1,self.embedding_dim)
            tail_type_compatibility = r_tt @ o_t.transpose(0,1)
        else:
            tail_type_compatibility = (o_t*r_tt).sum(-1)

        if flag_debug:
            print("Before Sigmoid")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))

        base_forward = torch.nn.Sigmoid()(self.psi*base_forward) + self.base_model_bias
        self.psi = 1.0
        head_type_compatibility = torch.nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = torch.nn.Sigmoid()(self.psi*tail_type_compatibility)

        if flag_debug:
            print("After Sigmoid")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))

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
        #"""

    def regularizer_icml(self, s, r, o):#, s_wt, r_wt, o_wt):
        #'''
        s_t = self.E_t(s)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o)
        #reg = (s_t*s_t + r_ht*r_ht + r_tt*r_tt + o_t*o_t).sum()
        #return self.base_model.regularizer(s, r, o) + reg
        
        factor = [torch.sqrt(s_t**2),torch.sqrt(o_t**2),torch.sqrt(r_ht**2+r_tt**2)]
        reg = 0
        for ele in factor:
            reg += torch.sum(torch.abs(ele) ** 3)

        #print("Prachi Debug","reg",reg.shape, reg, s.shape, reg/s.shape[0])
        #print("Prachi Debug","ele",ele.shape)
        return self.type_reg*(reg/s.shape[0]) + self.base_reg*(self.base_model.regularizer_icml(s, r, o))
        #'''
        #return self.base_model.regularizer_icml(s, r, o)


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

class typed_image_model_reg(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, image_compatibility_coefficient=0, image_embedding=None,flag_add_reverse=0):
        super(typed_image_model_reg, self).__init__()

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
        self.image_compatibility_coefficient = image_compatibility_coefficient
        self.image_embedding = torch.nn.Embedding(self.entity_count,image_embedding.shape[-1]) #TO DO incluse OOV vec
        #oov_random_embedding = numpy.random.rand(1,image_embedding.shape[-1])
        #image_embedding = numpy.vstack((image_embedding,oov_random_embedding))
        #self.image_embedding.weight.data.copy_(torch.from_numpy(image_embedding))

        self.image_embedding.weight.requires_grad = False

        self.linear = nn.Linear(image_embedding.shape[-1], self.embedding_dim)
        torch.nn.init.normal_(self.linear.weight.data, 0, 0.05)
        self.bn = nn.BatchNorm1d(self.embedding_dim, momentum=0.01)

        image_embedding = None
        print("IMPORTANT::TMP","added image-image compatibility!!")
    def forward(self, s, r, o, s_im, o_im, flag_debug=0):
        base_forward = self.base_model(s, r, o)
        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)
        if s is not None:
            tmp2 = self.image_embedding(s_im)
            tmp = self.linear(tmp2);
            tmp = tmp.view(-1,self.embedding_dim)
            s_image = self.bn(tmp)
            s_image = s_image.unsqueeze(1)
            if tmp2.shape[1]!=1:
                s_image = s_image.view(tmp2.shape[0],tmp2.shape[1],tmp.shape[1])
        else:
            tmp = self.linear(self.image_embedding.weight.unsqueeze(0))
            tmp = tmp.view(-1,self.embedding_dim)
            s_image = self.bn(tmp)
        
        if o is not None:
            tmp2 = self.image_embedding(o_im)
            tmp = self.linear(tmp2)
            tmp = tmp.view(-1,self.embedding_dim)
            o_image = self.bn(tmp)
            o_image = o_image.unsqueeze(1)
            if tmp2.shape[1]!=1:
                o_image = o_image.view(tmp2.shape[0],tmp2.shape[1],tmp.shape[1])
        else:
            tmp = self.linear(self.image_embedding.weight.unsqueeze(0));
            tmp = tmp.view(-1,self.embedding_dim)
            o_image = self.bn(tmp)
        
        head_type_compatibility_i = (s_image*r_ht).sum(-1)
        head_type_compatibility = (s_t*r_ht).sum(-1)
        image_head_type_compatibility = (s_t*s_image).sum(-1)
        #print("Prachi Debug","s_t,r_ht", s_t.shape, r_ht.shape)
        tail_type_compatibility_i = (o_image*r_tt).sum(-1)
        tail_type_compatibility = (o_t*r_tt).sum(-1)
        image_tail_type_compatibility = (o_t*s_image).sum(-1)

        base_forward = torch.nn.Sigmoid()(self.psi*base_forward)
        #head_type_compatibility_i = torch.nn.Sigmoid()(self.psi*head_type_compatibility_i)
        head_type_compatibility = torch.nn.Sigmoid()(self.psi*head_type_compatibility)
        #image_head_type_compatibility = torch.nn.Sigmoid()(self.psi*image_head_type_compatibility)
        #tail_type_compatibility_i = torch.nn.Sigmoid()(self.psi*tail_type_compatibility_i)
        tail_type_compatibility = torch.nn.Sigmoid()(self.psi*tail_type_compatibility)
        #image_tail_type_compatibility = torch.nn.Sigmoid()(self.psi*image_tail_type_compatibility)

        '''
        #all 3 like DM
        '''
        image_head = (s_image*r_ht*s_t).sum(dim=-1)        
        image_tail = (o_image*r_tt*o_t).sum(dim=-1)
        image_image = (s_image*o_image).sum(dim=-1)
 
        '''
        r_beta = self.R_beta(r)
        r_beta_val_tmp = self.linear_beta(r_beta) ;r_beta_val_tmp.view(r_beta_val_tmp.numel());r_beta_val = r_beta_val_tmp
        #r_beta_val_tmp.view(-1)
        #r_beta_val = self.bn_beta(r_beta_val_tmp)
        #r_beta_val.view(r_beta_val.numel())
        #print("Prachi Debug",r_beta[0],r_beta_val_tmp[0],r_beta_val[0]) 
        #print("base_forward",base_forward[5])
        #print("head_type_compatibility",head_type_compatibility[5])
        #print("tail_type_compatibility",tail_type_compatibility[5])
        #tmp = self.mult*base_forward*head_type_compatibility*tail_type_compatibility
        #print("Prachi Debug","final type model score",tmp.shape)
        '''

        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.5*(head_type_compatibility_i * tail_type_compatibility_i)) #51 - 1000
        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.005*(head_type_compatibility_i * tail_type_compatibility_i)) #45 - 500
        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.005*(head_type_compatibility_i * tail_type_compatibility_i) + 0.005*(image_head_type_compatibility * image_tail_type_compatibility) #44 - 500)
        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.005*(head_type_compatibility_i + tail_type_compatibility_i) + 0.005*(image_head_type_compatibility + image_tail_type_compatibility)#45.15)
        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.005*(head_type_compatibility_i + tail_type_compatibility_i)*(image_head_type_compatibility + image_tail_type_compatibility)#45.21 
        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.005*(head_type_compatibility_i * tail_type_compatibility_i * image_head_type_compatibility * image_tail_type_compatibility))#45.35
        

        ###return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.0005*(head_type_compatibility_i * tail_type_compatibility_i * image_head_type_compatibility * image_tail_type_compatibility))#45.85(500) - 57 (1k)   


        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.00005*(head_type_compatibility_i * tail_type_compatibility_i * image_head_type_compatibility * image_tail_type_compatibility))#45.34
        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.0005*((head_type_compatibility_i * image_head_type_compatibility) + (tail_type_compatibility_i * image_tail_type_compatibility)))#44.99(500) - 57.28(1k) 

        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.0005*(head_type_compatibility_i * image_head_type_compatibility) + 0.005*(tail_type_compatibility_i * image_tail_type_compatibility))#44 - 56

        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.00005*(image_head + image_tail))#wt linear-norm = 10 (w/t reg:36):: w/o = 45 (note that other img-ty model used linear norm) !!IMPORTANT!!
        ##yago coef = 0.000005 and fb15k 0.0005
        return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.005*(image_head + image_tail + image_image))#


        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + (image_head + image_tail))#9
        #return self.mult * base_forward * image_head * image_tail #1

        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.05*(head_type_compatibility_i * tail_type_compatibility_i * image_head_type_compatibility * image_tail_type_compatibility))#44.36 

        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + r_beta_val*(head_type_compatibility_i * tail_type_compatibility_i * image_head_type_compatibility * image_tail_type_compatibility)) 

    def regularizer(self, s, r, o):
        return self.base_model.regularizer(s, r, o)
    
    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            #self.linear.weight.data.div_(self.linear.weight.data.norm(2, dim=-1, keepdim=True))#!!!ATTENTION!!! - DM style thing works w/o this..rest work w/t this
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
            #self.R_beta.weight.data.div_(self.R_beta.weight.data.norm(2, dim=-1, keepdim=True))
            #self.linear_beta.weight.data.div_(self.linear_beta.weight.data.norm(2, dim=-1, keepdim=True))
            #test
            #self.image_embedding.weight.data.div_(self.image_embedding.weight.data.norm(2, dim=-1, keepdim=True))
            #test end

        return self.base_model.post_epoch()

    def image_compatibility(self, s,r, o, s_im, o_im):
        '''
        need to write a seperate function for evaluation scores
        -- as we plan to prestore all entity's image embeddings
        '''
        #print("Prachi Debug", s.shape, s_image.shape, o.shape, o_image.shape)
        s_t = self.E_t(s) #if s is not None else self.E_t.weight.unsqueeze(0)
        o_t = self.E_t(o) #if o is not None else self.E_t.weight.unsqueeze(0)

        tmp = self.linear(self.image_embedding(s_im));tmp = tmp.view(-1,self.embedding_dim)
        s_image = self.bn(tmp)
        tmp = self.linear(self.image_embedding(o_im));tmp = tmp.view(-1,self.embedding_dim)
        o_image = self.bn(tmp)

        s_image.unsqueeze_(1)
        o_image.unsqueeze_(1)

        s_image_compatibility = (s_t * s_image).sum(-1)
        o_image_compatibility = (o_t * o_image).sum(-1)
        image_image_compatibility = (s_image * o_image).sum(-1)
        
        s_image_compatibility.squeeze()
        o_image_compatibility.squeeze()
        image_image_compatibility.squeeze()

        return - s_image_compatibility, - o_image_compatibility, - image_image_compatibility#-(torch.mean(s_image_compatibility)), -(torch.mean(o_image_compatibility))

 

class typed_image_model(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, image_compatibility_coefficient=0, image_embedding=None,flag_add_reverse=0):
        super(typed_image_model, self).__init__()

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
        '''
        #beta model
        self.R_beta = torch.nn.Embedding(self.relation_count, 10)
        self.linear_beta = nn.Linear(10, 1)
        self.bn_beta = nn.BatchNorm1d(1, momentum=0.01)
        '''

        #image model
        self.image_compatibility_coefficient = image_compatibility_coefficient
        self.image_embedding = torch.nn.Embedding(image_embedding.shape[0],image_embedding.shape[-1]) #TO DO incluse OOV vec
        self.image_embedding.weight.data.copy_(torch.from_numpy(image_embedding))

        self.image_embedding.weight.requires_grad = False##

        self.linear = nn.Linear(image_embedding.shape[-1], self.embedding_dim)
        torch.nn.init.normal_(self.linear.weight.data, 0, 0.05)
        self.bn = nn.BatchNorm1d(self.embedding_dim, momentum=0.01)

        image_embedding = None

        print("Beta model")
        '''
        self.beta_head = torch.nn.Embedding(self.relation_count, 1)
        torch.nn.init.constant_(self.beta_head.weight.data, -5.0)

        self.beta_tail = torch.nn.Embedding(self.relation_count, 1)
        torch.nn.init.constant_(self.beta_tail.weight.data, -5.0)

        self.beta_img = torch.nn.Embedding(self.relation_count, 1)
        torch.nn.init.constant_(self.beta_img.weight.data, -5.0)
        '''
        print("IMPORTANT::TMP","added image-image compatibility!!")
    def forward(self, s, r, o, s_im, o_im, flag_debug=0):
        base_forward = self.base_model(s, r, o)
        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)
        '''#mrr: 2.25
        tmp = self.image_embedding(s) if s is not None else self.image_embedding.weight.unsqueeze(0)
        s_image =  self.linear(tmp)

        tmp = self.image_embedding(o) if o is not None else self.image_embedding.weight.unsqueeze(0)
        o_image =  self.linear(tmp)
        '''
        ##with bn -- 9 mrr
        if 1:#s is not None:
            tmp2 = self.image_embedding(s_im)
            tmp = self.linear(tmp2);
            tmp = tmp.view(-1,self.embedding_dim)
            s_image = self.bn(tmp)
            s_image = s_image.unsqueeze(1)
            if tmp2.shape[1]!=1:
                s_image = s_image.view(tmp2.shape[0],tmp2.shape[1],tmp.shape[1])
        else:
            s_image = 0
          
        
        if 1:#o is not None:
            tmp2 = self.image_embedding(o_im)
            tmp = self.linear(tmp2)
            tmp = tmp.view(-1,self.embedding_dim)
            o_image = self.bn(tmp)
            o_image = o_image.unsqueeze(1)
            if tmp2.shape[1]!=1:
                o_image = o_image.view(tmp2.shape[0],tmp2.shape[1],tmp.shape[1])
        else:
            o_image = 0
        '''
        print("Ready!!:")
        print("Prachi Debug","o_t",o_t.shape)
        print("Prachi Debug","s_t",s_t.shape)
        print("Prachi Debug","o_image",o_image.shape)
        print("Prachi Debug","s_image",s_image.shape) 
        '''
        #if not(s_image.shape[0] == r_ht.shape[0]):#s is None:
        #    s_image = s_image.view(s_image.shape[1], s_image.shape[0], s_image.shape[2])
            #s_t = s_t.view(1, s_t.shape[0], s_t.shape[1])
            #oov_vec = s_t[0][-1]; oov_vec_pad = oov_vec.repeat(s_image.shape[1]-s_t.shape[1],1)
            #oov_vec_pad = oov_vec_pad.unsqueeze(0)
            #s_t = torch.cat([s_t,oov_vec_pad],dim=1)

        '''
        head_type_compatibility_i = (s_image*r_ht).sum(-1)
        '''
        head_type_compatibility = (s_t*r_ht).sum(-1)
        '''
        image_head_type_compatibility = (s_t*s_image).sum(-1)
        print("Prachi Debug","s_t,r_ht", s_t.shape, r_ht.shape)
        print("Prachi Debug","s_image",s_image.shape)
        print("Prachi Debug","o_image",o_image.shape)
        print("Prachi Debug","r_tt",r_tt.shape)
        print("Prachi Debug","o_t",o_t.shape)
        print("Prachi Debug","image_head_type_compatibility",image_head_type_compatibility.shape)
        print("Prachi Debug","head_type_compatibility",head_type_compatibility.shape)
        print("Prachi Debug","head_type_compatibility_i",head_type_compatibility_i.shape)
        tail_type_compatibility_i = (o_image*r_tt).sum(-1)
        '''
        tail_type_compatibility = (o_t*r_tt).sum(-1)
        '''
        image_tail_type_compatibility = (o_t*s_image).sum(-1)
        '''
        base_forward = torch.nn.Sigmoid()(self.psi*base_forward)
        #head_type_compatibility_i = torch.nn.Sigmoid()(self.psi*head_type_compatibility_i)
        head_type_compatibility = torch.nn.Sigmoid()(self.psi*head_type_compatibility)
        #image_head_type_compatibility = torch.nn.Sigmoid()(self.psi*image_head_type_compatibility)
        #tail_type_compatibility_i = torch.nn.Sigmoid()(self.psi*tail_type_compatibility_i)
        tail_type_compatibility = torch.nn.Sigmoid()(self.psi*tail_type_compatibility)
        #image_tail_type_compatibility = torch.nn.Sigmoid()(self.psi*image_tail_type_compatibility)
        '''
        #all 3 like DM
        '''
        '''
        image_head = (s_image*r_ht*s_t).sum(dim=-1)        
        image_tail = (o_image*r_tt*o_t).sum(dim=-1)
        
        image_image = (o_image*s_image).sum(dim=-1)
        '''
        image_head = torch.nn.Sigmoid()(s_image*r_ht*s_t).sum(dim=-1)        
        image_tail = torch.nn.Sigmoid()(o_image*r_tt*o_t).sum(dim=-1)
        image_image = torch.nn.Sigmoid()(o_image*s_image).sum(dim=-1)
 
        '''
        r_beta = self.R_beta(r)
        r_beta_val_tmp = self.linear_beta(r_beta) ;r_beta_val_tmp.view(r_beta_val_tmp.numel());r_beta_val = r_beta_val_tmp
        #r_beta_val_tmp.view(-1)
        #r_beta_val = self.bn_beta(r_beta_val_tmp)
        #r_beta_val.view(r_beta_val.numel())
        #print("Prachi Debug",r_beta[0],r_beta_val_tmp[0],r_beta_val[0]) 
        #print("base_forward",base_forward[5])
        #print("head_type_compatibility",head_type_compatibility[5])
        #print("tail_type_compatibility",tail_type_compatibility[5])
        #tmp = self.mult*base_forward*head_type_compatibility*tail_type_compatibility
        #print("Prachi Debug","final type model score",tmp.shape)
        '''
        '''
        beta_head_val = self.beta_head(r).squeeze(2)
        beta_tail_val = self.beta_tail(r)
        beta_img_val = self.beta_img(r)
        #print("Prachi Debug","beta_head_val",beta_head_val[:10])
        #print("Prachi Debug","beta_tail_val",beta_tail_val[:10])
        #print("Prachi Debug","beta_img_val",beta_img_val[:10])
        beta_head_val = torch.nn.Sigmoid()(beta_head_val)
        beta_tail_val = torch.nn.Sigmoid()(beta_tail_val)
        beta_img_val = torch.nn.Sigmoid()(beta_img_val)
        print("after")
        print("Prachi Debug","beta_head_val",beta_head_val[:10])
        print("Prachi Debug","beta_tail_val",beta_tail_val[:10])
        #print("Prachi Debug","beta_img_val",beta_img_val[:10])
        print("Prachi Debug","base_forward",base_forward[:10])
        print("Prachi Debug","head_type_compatibility",head_type_compatibility[:10])
        print("Prachi Debug","image_head",image_head[:10])
        print("Prachi Debug","image_tail",image_tail[:10])
        score = self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + (beta_head_val*image_head + beta_tail_val*image_tail))# + beta_img_val*image_image))'''
        score = self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.005*(image_head) + 0.005*(image_tail))#2# + 0.005*(image_image)) #1
        #print("Prachi Debug","score",score.shape)
        return score

        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.5*(head_type_compatibility_i * tail_type_compatibility_i)) #51 - 1000
        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.005*(head_type_compatibility_i * tail_type_compatibility_i)) #45 - 500
        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.005*(head_type_compatibility_i * tail_type_compatibility_i) + 0.005*(image_head_type_compatibility * image_tail_type_compatibility) #44 - 500)
        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.005*(head_type_compatibility_i + tail_type_compatibility_i) + 0.005*(image_head_type_compatibility + image_tail_type_compatibility)#45.15)
        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.005*(head_type_compatibility_i + tail_type_compatibility_i)*(image_head_type_compatibility + image_tail_type_compatibility)#45.21 
        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.005*(head_type_compatibility_i * tail_type_compatibility_i * image_head_type_compatibility * image_tail_type_compatibility))#45.35
        

        ###return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.0005*(head_type_compatibility_i * tail_type_compatibility_i * image_head_type_compatibility * image_tail_type_compatibility))#45.85(500) - 57 (1k)   


        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.00005*(head_type_compatibility_i * tail_type_compatibility_i * image_head_type_compatibility * image_tail_type_compatibility))#45.34
        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.0005*((head_type_compatibility_i * image_head_type_compatibility) + (tail_type_compatibility_i * image_tail_type_compatibility)))#44.99(500) - 57.28(1k) 

        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.0005*(head_type_compatibility_i * image_head_type_compatibility) + 0.005*(tail_type_compatibility_i * image_tail_type_compatibility))#44 - 56

        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.0005*(image_head + image_tail))#wt linear-norm = 10 (w/t reg:36):: w/o = 45 (note that other img-ty model used linear norm) !!IMPORTANT!!

        #####return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.005*(image_head + image_tail + image_image))#wt linear-norm = 10 (w/t reg:36):: w/o = 45 (note that other img-ty model used linear norm) !!IMPORTANT!!

        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + (image_head + image_tail))#9
        #return self.mult * base_forward * image_head * image_tail #1

        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + 0.05*(head_type_compatibility_i * tail_type_compatibility_i * image_head_type_compatibility * image_tail_type_compatibility))#44.36 

        #return self.mult * ((base_forward * head_type_compatibility * tail_type_compatibility) + r_beta_val*(head_type_compatibility_i * tail_type_compatibility_i * image_head_type_compatibility * image_tail_type_compatibility)) 

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
            #self.linear.weight.data.div_(self.linear.weight.data.norm(2, dim=-1, keepdim=True))#!!!ATTENTION!!! - DM style thing works w/o this..rest work w/t this
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
            #self.R_beta.weight.data.div_(self.R_beta.weight.data.norm(2, dim=-1, keepdim=True))
            #self.linear_beta.weight.data.div_(self.linear_beta.weight.data.norm(2, dim=-1, keepdim=True))
            #test
            #self.image_embedding.weight.data.div_(self.image_embedding.weight.data.norm(2, dim=-1, keepdim=True))
            #test end

        return self.base_model.post_epoch()

    def image_compatibility(self, s,r, o, s_im, o_im):
        '''
        need to write a seperate function for evaluation scores
        -- as we plan to prestore all entity's image embeddings
        '''
        #print("Prachi Debug", s.shape, s_image.shape, o.shape, o_image.shape)
        s_t = self.E_t(s) #if s is not None else self.E_t.weight.unsqueeze(0)
        o_t = self.E_t(o) #if o is not None else self.E_t.weight.unsqueeze(0)

        tmp = self.linear(self.image_embedding(s_im));tmp = tmp.view(-1,self.embedding_dim)
        s_image = self.bn(tmp)
        tmp = self.linear(self.image_embedding(o_im));tmp = tmp.view(-1,self.embedding_dim)
        o_image = self.bn(tmp)

        s_image.unsqueeze_(1)
        o_image.unsqueeze_(1)

        s_image_compatibility = (s_t * s_image).sum(-1)
        o_image_compatibility = (o_t * o_image).sum(-1)
        image_image_compatibility = (s_image * o_image).sum(-1)

        s_image_compatibility.squeeze()
        o_image_compatibility.squeeze()
        image_image_compatibility.squeeze()

        return - s_image_compatibility, - o_image_compatibility, - image_image_compatibility#-(torch.mean(s_image_compatibility)), -(torch.mean(o_image_compatibility))


class only_image_model(torch.nn.Module):
    #replace s,o type embedding w/t image embeddings
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, image_compatibility_coefficient=0, image_embedding=None):
        super(only_image_model, self).__init__()

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
        ##self.E_t = torch.nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_ht = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tt = torch.nn.Embedding(self.relation_count, self.embedding_dim)
        ##torch.nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        torch.nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        self.minimum_value = 0.0

        #image model
        self.image_compatibility_coefficient = image_compatibility_coefficient
        #self.image_model = EncoderCNN(self.embedding_dim)
        ##self.image_embedding = torch.nn.Embedding(self.entity_count-1,image_embedding.shape[-1]) #TO DO incluse OOV vec
        self.image_embedding = torch.nn.Embedding(self.entity_count,image_embedding.shape[-1]) #TO DO incluse OOV vec
        oov_random_embedding = numpy.random.rand(1,image_embedding.shape[-1])
        image_embedding = numpy.vstack((image_embedding,oov_random_embedding))
        self.image_embedding.weight.data.copy_(torch.from_numpy(image_embedding))

        self.image_embedding.weight.requires_grad = False

        self.linear = nn.Linear(image_embedding.shape[-1], self.embedding_dim)
        torch.nn.init.normal_(self.linear.weight.data, 0, 0.05)
        self.bn = nn.BatchNorm1d(self.embedding_dim, momentum=0.01)

        image_embedding = None

    def forward(self, s, r, o, flag_debug=0):
        base_forward = self.base_model(s, r, o)
        #s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        #o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)
        '''#mrr: 2.25
        tmp = self.image_embedding(s) if s is not None else self.image_embedding.weight.unsqueeze(0)
        s_image =  self.linear(tmp)

        tmp = self.image_embedding(o) if o is not None else self.image_embedding.weight.unsqueeze(0)
        o_image =  self.linear(tmp)
        '''
        ##with bn -- 9 mrr
        if s is not None:
            tmp2 = self.image_embedding(s)
            tmp = self.linear(tmp2);
            tmp = tmp.view(-1,self.embedding_dim)
            s_image = self.bn(tmp)
            s_image = s_image.unsqueeze(1)
            if tmp2.shape[1]!=1:
                s_image = s_image.view(tmp2.shape[0],tmp2.shape[1],tmp.shape[1])
        else:
            tmp = self.linear(self.image_embedding.weight.unsqueeze(0))
            tmp = tmp.view(-1,self.embedding_dim)
            s_image = self.bn(tmp)
        
        if o is not None:
            tmp2 = self.image_embedding(o)
            tmp = self.linear(tmp2)
            tmp = tmp.view(-1,self.embedding_dim)
            o_image = self.bn(tmp)
            o_image = o_image.unsqueeze(1)
            if tmp2.shape[1]!=1:
                o_image = o_image.view(tmp2.shape[0],tmp2.shape[1],tmp.shape[1])
        else:
            tmp = self.linear(self.image_embedding.weight.unsqueeze(0));
            tmp = tmp.view(-1,self.embedding_dim)
            o_image = self.bn(tmp)
        
        head_type_compatibility = (s_image*r_ht).sum(-1)#(s_t*r_ht).sum(-1)
        #print("Prachi Debug","s_t,r_ht", s_t.shape, r_ht.shape)
        tail_type_compatibility = (o_image*r_tt).sum(-1)#(o_t*r_tt).sum(-1)

        base_forward = torch.nn.Sigmoid()(self.psi*base_forward)
        head_type_compatibility = torch.nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = torch.nn.Sigmoid()(self.psi*tail_type_compatibility)
        #print("base_forward",base_forward[5])
        #print("head_type_compatibility",head_type_compatibility[5])
        #print("tail_type_compatibility",tail_type_compatibility[5])
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
    
    def post_epoch(self):
        if(self.unit_reg):
            #self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.linear.weight.data.div_(self.linear.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
            #test
            #self.image_embedding.weight.data.div_(self.image_embedding.weight.data.norm(2, dim=-1, keepdim=True))
            #test end

        return self.base_model.post_epoch()


class image_model(torch.nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, image_compatibility_coefficient=0, image_embedding=None):
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
        self.image_compatibility_coefficient = image_compatibility_coefficient
        #self.image_model = EncoderCNN(self.embedding_dim)
        self.image_embedding = torch.nn.Embedding(image_embedding.shape[0],image_embedding.shape[-1]) #TO DO incluse OOV vec
        self.image_embedding.weight.data.copy_(torch.from_numpy(image_embedding))

        self.image_embedding.weight.requires_grad = False##
        print("TEMP!!!!!!!!!!!!!SWITCHING OFF image embed finetuning!!!!!!!")

        self.linear = nn.Linear(image_embedding.shape[-1], self.embedding_dim)
        torch.nn.init.normal_(self.linear.weight.data, 0, 0.05)##

        self.bn = nn.BatchNorm1d(self.embedding_dim, momentum=0.01)

        image_embedding = None

    def forward(self, s, r, o, s_im, o_im, flag_debug=0):
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

    def image_compatibility(self, s,r, o, s_im, o_im):
        '''
        need to write a seperate function for evaluation scores
        -- as we plan to prestore all entity's image embeddings
        '''
        #print("Prachi Debug", s.shape, s_image.shape, o.shape, o_image.shape)
        s_t = self.E_t(s) #if s is not None else self.E_t.weight.unsqueeze(0)
        o_t = self.E_t(o) #if o is not None else self.E_t.weight.unsqueeze(0)

        ##
        #r_ht = self.R_ht(r)
        #r_tt = self.R_tt(r)
        ##

        tmp = self.linear(self.image_embedding(s_im));tmp = tmp.view(-1,self.embedding_dim)
        s_image = self.bn(tmp)
        tmp = self.linear(self.image_embedding(o_im));tmp = tmp.view(-1,self.embedding_dim)
        o_image = self.bn(tmp)

        #s_image = torch.nn.Sigmoid()(s_image_1)
        #o_image = torch.nn.Sigmoid()(o_image_1)

        #print("Prachi Debug","s_image.shape",s_image.shape)
        #print("Prachi Debug","s_t.shape",s_t.shape)
        s_image.unsqueeze_(1)
        o_image.unsqueeze_(1)

        #print("Prachi Debug","s_image.shape 2",s_image.shape)

        #s_image_compatibility = (s_image * s_image).sum(-1)
        s_image_compatibility = (s_t * s_image).sum(-1)
        #o_image_compatibility = (o_image * o_image).sum(-1)
        o_image_compatibility = (o_t * o_image).sum(-1)
        image_image_compatibility = (s_image * o_image).sum(-1) 
        #print("Prachi Debug","s_image_compatibility.shape",s_image_compatibility.shape)
        ##
        #s_image_compatibility = torch.nn.Sigmoid()(self.psi*s_image_compatibility)
        #o_image_compatibility = torch.nn.Sigmoid()(self.psi*o_image_compatibility)
        #print("Prachi Debug","s_image_compatibility.shape", s_image_compatibility.shape)

        ##
        #s_r_image_compatibility = (r_ht * s_image).sum(-1)
        #o_r_image_compatibility = (r_tt * o_image).sum(-1)
        #o_r_image_compatibility.squeeze()
        #s_r_image_compatibility.squeeze()
        ##

        s_image_compatibility.squeeze()
        o_image_compatibility.squeeze()

        #print("Prachi Debug","s_image_compatibility.shape 2", s_image_compatibility.shape)

        return - s_image_compatibility, - o_image_compatibility, - image_image_compatibility#-(torch.mean(s_image_compatibility)), -(torch.mean(o_image_compatibility))#, -(torch.mean(s_r_image_compatibility)), -(torch.mean(o_r_image_compatibility))

    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
            # to test
            #self.linear.weight.data.div_(self.linear.weight.data.norm(2, dim=-1, keepdim=True))
            #failed
            #self.image_embedding.weight.data.div_(self.image_embedding.weight.data.norm(2, dim=-1, keepdim=True))
            #test end

        return self.base_model.post_epoch()


class DME(torch.nn.Module):
    """
    DM+E model.
    deprecated. Use Adder model with DM and E as sub models for more control
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False,flag_add_reverse=0):
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
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=True, clamp_v=None, display_norms=False, flag_add_reverse=0):
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
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, mult=20.0, box_reg_coef=0.1, box_reg='l2', psi=2.0,flag_add_reverse=0):
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
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, mult=20.0, box_reg_coef=0.1, box_reg='l2', psi=2.0, flag_add_reverse=0):
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
