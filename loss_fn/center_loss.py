import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from .triplet_loss import *


class CenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, size_average=True):

        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average
        self.triplet= TripletLoss(margin=0.3)


    def forward(self, label, feat):

        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)


        loss = self.centerlossfunc(feat, label, F.normalize(self.centers), batch_size_tensor)

        return loss


class CenterlossFunc(Function):

    @staticmethod

    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        m=nn.ReLU(inplace=False)
        margin=1
        out=(feature - centers_batch).pow(2).sum(1)-margin
        out=m(torch.exp(out)-1)
        # out=(feature - centers_batch).pow(2).sum()
        # tri_loss, prec,dist_ap,dist_an=triplet(centers_batch,centers_batch,label)
        center_loss=out.sum() / 2.0 / batch_size
        # ap_loss=torch.mean(-torch.log(dist_ap)-torch.log(dist_an))
        all_loss=center_loss
        return all_loss

    @staticmethod

    def backward(ctx, grad_output):

        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        m = nn.ReLU(inplace=False)
        diff = centers_batch - feature#

        diff=torch.mul(torch.exp(feature),diff)
        # diff = torch.mul(torch.exp(diff-0.05), diff)
        # diff = torch.exp(m(diff.pow(2).sum(1) - 0.1)-1).reshape((64,1))* diff
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


class CenterLoss_cro(nn.Module):

    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss_cro, self).__init__()
        self.centers1 = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centers2 = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc_cro.apply
        self.feat_dim = feat_dim
        self.size_average = size_average
        self.triplet = TripletLoss(margin=0.3)

    def forward(self, label1,label2, feat1,feat2):
        batch_size = feat1.size(0)
        feat1 = feat1.view(batch_size, -1)
        feat2 = feat2.view(batch_size, -1)
        # To check the dim of centers and features
        if feat1.size(1) != self.feat_dim:
            raise ValueError(
                "Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim, feat1.size(1)))
        batch_size_tensor = feat1.new_empty(1).fill_(batch_size if self.size_average else 1)

        loss = self.centerlossfunc(feat1,feat2, label1,label2, self.centers1, self.centers2,batch_size_tensor)

        return loss



class CenterlossFunc_cro(Function):

    @staticmethod

    def forward(ctx, feature1, feature2,label1, label2,centers1, centers2,batch_size):
        ctx.save_for_backward(feature1, feature2,label1, label2, centers1, centers2,batch_size)
        centers_batch1 = centers1.index_select(0, label1.long())
        centers_batch2 = centers2.index_select(0, label2.long())
        m=nn.ReLU(inplace=False)
        margin=0.05
        out1=m((feature1 - centers_batch1).pow(2).sum(1)-margin)
        center_loss1=out1.sum() / 2.0 / batch_size
        out2 = m((feature2 - centers_batch2).pow(2).sum(1) - margin)
        center_loss2 = out2.sum() / 2.0 / batch_size
        out3=m((feature1- centers_batch2).pow(2).sum(1) - margin)
        center_loss3 = out3.sum() / 2.0 / batch_size
        out4 = m((feature2 - centers_batch1).pow(2).sum(1) - margin)
        center_loss4 = out4.sum() / 2.0 / batch_size
        all_loss=(center_loss1+center_loss2+center_loss3+center_loss4)/4.0
        return all_loss

    @staticmethod

    def backward(ctx, grad_output):

        sigma=2.0
        feature1, feature2,label1, label2, centers1, centers2,batch_size = ctx.saved_tensors
        centers_batch1 = centers1.index_select(0, label1.long())
        diff1 = centers_batch1 - feature1#here add centers_batch(centers_batch - feature)/sigma
        centers_batch2 = centers2.index_select(0, label2.long())
        diff2 = centers_batch2 - feature2
        # init every iteration
        counts = centers1.new_ones(centers1.size(0))
        ones = centers1.new_ones(label1.size(0))
        counts = counts.scatter_add_(0, label1.long(), ones)

        grad_centers1 = centers1.new_zeros(centers1.size())
        grad_centers1.scatter_add_(0, label1.unsqueeze(1).expand(feature1.size()).long(), diff1)
        grad_centers1 = grad_centers1/counts.view(-1, 1)
        grad_centers2 = centers2.new_zeros(centers2.size())
        grad_centers2.scatter_add_(0, label2.unsqueeze(1).expand(feature2.size()).long(), diff2)
        grad_centers2 = grad_centers2 / counts.view(-1, 1)
        return - grad_output * diff1 / batch_size, - grad_output * diff2 / batch_size,None, None,grad_centers1 / batch_size, grad_centers2 / batch_size,None

