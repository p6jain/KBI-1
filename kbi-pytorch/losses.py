import torch
import torch.nn as nn

class crossentropy_loss(torch.nn.Module):
    def __init__(self):
        super(crossentropy_loss, self).__init__()
        self.name = "crossentropy_loss"
        self.loss = nn.CrossEntropyLoss(reduction = 'mean')
    def forward(self, scores, ans):
        ans = ans.view(ans.shape[0])
        losses = self.loss(scores, ans)
        return losses


class crossentropy_lossV2(torch.nn.Module):
    def __init__(self):
        super(crossentropy_lossV2, self).__init__()
        self.name = "crossentropy_lossV2"
        self.loss1 = nn.LogSoftmax(dim=1)
        self.loss2 = nn.NLLLoss()
    def forward(self, scores, ans):
        ans = ans.view(ans.shape[0])
        print("SCORES:BS",scores,scores.shape)
        scores = self.loss1(scores)
        print("SCORES: AS",scores,scores.shape)
        losses = self.loss2(scores, ans)
        print("LOSS",losses,losses.shape)
        return losses


class softmax_loss(torch.nn.Module):
    def __init__(self):
        super(softmax_loss, self).__init__()
        self.name = "softmax_loss"
    def forward(self, positive, negative_1, negative_2):
        max_den_e1 = negative_1.max(dim=1, keepdim=True)[0].detach()
        max_den_e2 = negative_2.max(dim=1, keepdim=True)[0].detach()
        den_e1 = (negative_1-max_den_e1).exp().sum(dim=-1, keepdim=True)
        den_e2 = (negative_2-max_den_e2).exp().sum(dim=-1, keepdim=True)
        losses = ((2*positive-max_den_e1-max_den_e2) - den_e1.log() - den_e2.log())
        '''
        print("Prachi debug","negative_1",negative_1[:10])
        print("Prachi debug","negative_2",negative_2[:10])
        print("Prachi debug","den_e1",den_e1[:10],den_e1.log()[:10])
        print("Prachi debug","den_e2",den_e2[:10],den_e2.log()[:10])
        print("Prachi debug","positive-max_den_e1",positive-max_den_e1)
        print("Prachi debug","pos-max_den_e2",positive-max_den_e2)
        print("Prachi debug","losses",losses)'''
        return -losses.mean()


class softmax_loss_reductionMean(torch.nn.Module):
    def __init__(self):
        super(softmax_loss_reductionMean, self).__init__()
        self.name = "softmax_loss_reductionMean"
    def forward(self, positive, negative_1, negative_2):
        max_den_e1 = negative_1.max(dim=1, keepdim=True)[0].detach()
        max_den_e2 = negative_2.max(dim=1, keepdim=True)[0].detach()
        den_e1 = (negative_1-max_den_e1).exp().mean(dim=-1, keepdim=True)
        den_e2 = (negative_2-max_den_e2).exp().mean(dim=-1, keepdim=True)
        losses = ((2*positive-max_den_e1-max_den_e2) - den_e1.log() - den_e2.log())
        return -losses.mean()


class logistic_loss(torch.nn.Module):
    def __init__(self):
        super(logistic_loss, self).__init__()
        self.name = "logistic_loss"
    def forward(self, positive, negative_1, negative_2):
        scores = torch.cat([positive, negative_1, negative_2], dim=-1)
        truth = torch.ones(1, positive.shape[1]+negative_1.shape[1]+negative_2.shape[1]).cuda()
        truth[0, 0] = -1
        truth = -truth
        truth = torch.autograd.Variable(truth, requires_grad=False)
        x = torch.log(1+torch.exp(-scores*truth))
        total = x.sum()
        return total/((positive.shape[1]+negative_1.shape[1]+negative_2.shape[1])*positive.shape[0])

