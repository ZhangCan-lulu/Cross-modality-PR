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


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.m=nn.ReLU(inplace=False)
    def forward(self, inputs1, inputs2,targets):
        n = inputs1.size(0)
        # #print('n:',n)
        # # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs1, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs1, inputs2.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        # shape [N, N]
        global_feat1 = normalize(inputs1, axis=-1)
        global_feat2 = normalize(inputs2, axis=-1)
        # shape [N, N]

        dist = euclidean_dist(global_feat1, global_feat2)
        # dist = cosin_dist(global_feat1, global_feat2)
        # dist =F.cosine_similarity(global_feat1, global_feat2,1)
        is_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        is_neg = targets.expand(n, n).ne(targets.expand(n, n).t())

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap, relative_p_inds = torch.max(
            dist[is_pos].contiguous().view(n, -1), 1, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an, relative_n_inds = torch.min(
            dist[is_neg].contiguous().view(n, -1), 1, keepdim=True)
        # shape [N]
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        #print('dist_ap:',dist_ap)
    # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # out=dist_ap+self.margin-dist_an
        # out = self.m(torch.exp(out) - 1)
        # loss=out.mean()
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec,dist_ap,dist_an

    def dist_apn(self, dist_an, dist_ap):
        """
        Args:
          dist_ap: pytorch Variable, distance between anchor and positive sample,
            shape [N]
          dist_an: pytorch Variable, distance between anchor and negative sample,
            shape [N]
        Returns:
          loss: pytorch Variable, with shape [1]
        """
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        if self.margin is not None:
          loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
          loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss

    def dist_l2_apn(self, feat1, feat2):
        batch_size=feat1.size(0)
        m = nn.ReLU(inplace=False)
        margin = 0.1
        out = m((feat1 - feat2).pow(2).sum(1) - margin)
        center_loss = out.sum() / 2.0 / batch_size
        return center_loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss