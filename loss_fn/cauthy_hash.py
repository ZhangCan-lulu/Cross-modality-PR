from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosin_dist(x,y):
    """
        Args:
          x: pytorch Variable, with shape [m, d]
          y: pytorch Variable, with shape [n, d]
        Returns:
          dist: pytorch Variable, with shape [m, n]
     """
    # m, n = x.size(0), y.size(0)
    x_n=torch.norm(x,2,1,keepdim=True).expand_as(x)
    y_n=torch.norm(x,2,1,keepdim=True).expand_as(y)

    x_l=torch.div(x,x_n)
    y_l=torch.div(y,y_n).t()
    dist=torch.mm(x_l,y_l).clamp(min=1e-12)

    return dist


def cauchy_cross_entropy(inputs1, target1,inputs2=None, target2=None, gamma=1, normed=True):
    if inputs1 is None:
        inputs2 = inputs1
        target2 = target1
    n=inputs1.size(0)
    global_feat1 = normalize(inputs1, axis=-1)
    global_feat2 = normalize(inputs2, axis=-1)
    # shape [N, N]

    dist = euclidean_dist(global_feat1, global_feat2)
    cauchy = gamma / (dist + gamma)
    # dist =F.cosine_similarity(global_feat1, global_feat2,1)
    is_pos = target1.expand(n, n).eq(target2.expand(n, n).t())
    is_neg = target1.expand(n, n).ne(target2.expand(n, n).t())

    cauchy_mask =torch.ones_like(cauchy)
    cauchy_mask[is_pos] = cauchy[is_pos]

    all_loss = - is_pos * \
               torch.log(cauchy_mask) - (1.0 - is_pos) * \
               torch.log(1.0 - cauchy_mask)

    sum_all=torch.sum(is_pos)+torch.sum(is_neg)
    sum_l=torch.sum(is_pos)
    balance_p_mask=torch.add(torch.abs(torch.add(is_pos,-1.0)),torch.mul(torch.div(sum_all,sum_l)))
    cro_loss=torch.mean(torch.mul(all_loss,balance_p_mask))
    return cro_loss

class FL_loss(nn.Module):

    def __init__(self, gamma=0):
        super(FL_loss, self).__init__()
        self.gamma = gamma

    def forward(self,p):
        logp = torch.log(p)
        loss = -(1 - p) ** self.gamma * logp
        return loss.mean()

def triplet_cross_sigma(inputs1, target1,inputs2=None, target2=None,num_class=395):
    if inputs2 is None:
        inputs2 = inputs1
        target2 = target1
    n=inputs1.size(0)
    global_feat1 = normalize(inputs1, axis=-1)
    global_feat2 = normalize(inputs2, axis=-1)
    dist = euclidean_dist(global_feat1, global_feat2)
    # dist=cosin_dist(global_feat1, global_feat2)
    is_pos = target1.expand(n, n).eq(target2.expand(n, n).t())
    is_neg = target1.expand(n, n).ne(target2.expand(n, n).t())

    dist_ap, relative_p_inds = torch.max(
        dist[is_pos].contiguous().view(n, -1), 1, keepdim=True)

    dist_an, relative_n_inds = torch.min(
        dist[is_neg].contiguous().view(n, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    margin=10
    # x=torch.mul(torch.log(torch.div(dist_an,dist_ap)),margin)
    x=dist_an-dist_ap
    prob=torch.sigmoid(margin*x)

    out=torch.ones(n,num_class)
    CE= nn.CrossEntropyLoss().to(device)
    # FL=FL_loss(gamma=5)
    # loss=FL(prob)
    out_pro=-torch.log(prob)
    rank=torch.nn.MarginRankingLoss()
    y = dist_an.data.new()
    y.resize_as_(dist_an.data)
    y.fill_(1)
    y = Variable(y)

    ap_input = dist_ap.data.new()
    ap_input.resize_as_(dist_ap.data)
    ap_input.fill_(0.3)
    ap_input = Variable(ap_input)
    rank_loss=rank(ap_input,dist_ap,y)
    loss=torch.mean(out_pro)+rank_loss
    return loss

def cross_tri_sigma_all(inputs1, target1, inputs2, target2):


    inter_cro_v=triplet_cross_sigma(inputs1, target1, inputs2, target2)
    inter_cro_t = triplet_cross_sigma(inputs2, target2, inputs1, target1)

    intra_cro_v=triplet_cross_sigma(inputs1, target1)
    intra_cro_t = triplet_cross_sigma(inputs2, target2)

    loss=inter_cro_t+inter_cro_v+intra_cro_t+intra_cro_v
    return loss