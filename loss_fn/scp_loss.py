from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from .center_loss import *
from .OIM_loss import *

class TripletLoss_id(nn.Module):

    def __init__(self, margin=None):
        super(TripletLoss_id, self).__init__()
        self.margin = margin
        if margin is not None:
          self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
          self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, dist_an, dist_ap):
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



def hard_example_mining(dist_mat, labels1,labels2,return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels1.expand(N, N).eq(labels2.expand(N, N).t())
    is_neg = labels1.expand(N, N).ne(labels2.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels1.new().resize_as_(labels1)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def global_loss(tri_loss, global_feat1, global_feat2,labels1,labels2,normalize_feature=True):
    """
    Args:
      tri_loss: a `TripletLoss` object
      global_feat: pytorch Variable, shape [N, C]
      labels: pytorch LongTensor, with shape [N]
      normalize_feature: whether to normalize feature to unit length along the
        Channel dimension
    Returns:
      loss: pytorch Variable, with shape [1]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
      ==================
      For Debugging, etc
      ==================
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
    """
    if normalize_feature:
        global_feat1 = normalize(global_feat1, axis=-1)
        global_feat2 = normalize(global_feat2, axis=-1)
    # shape [N, N]
    dist_mat = euclidean_dist(global_feat1, global_feat2)
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
        dist_mat, labels1,labels2,return_inds=True)
    loss = tri_loss(dist_an, dist_ap)
    prec = (dist_an.data > dist_ap.data).sum() * 1. / labels1.size(0)
    return [loss, p_inds, n_inds, dist_ap, dist_an, dist_mat,prec]




def triplet_hard_loss(feat1,feat2, P, K):
    sh0 = feat1.size(0)  # number of samples
    sh1 = feat1.size(1)  # feature dimension
    assert P * K == sh0, "Error: Dimension does not match! P={},K={},sh0={}".format(P, K, sh0)
    feat1 = feat1.view(sh0, 1, sh1).repeat(1, sh0, 1)
    feat2 = feat2.view(1, sh0, sh1).repeat(sh0, 1, 1)
    delta = feat1 - feat2
    l2 = torch.sqrt((delta * delta).sum(dim=2) + 1e-8)
    positive = [l2[i * K:i * K + K, i * K:i * K + K] for i in range(P)]
    positive = torch.cat(positive, dim=0)
    positive, _ = positive.max(dim=1)
    negative = []
    for i in range(P):
        tmp = [l2[i * K:i * K + K, j * K:j * K + K] for j in range(P) if i != j]
        tmp = torch.cat(tmp, dim=1)
        negative.append(tmp)
    negative = torch.cat(negative, dim=0)
    negative, _ = negative.min(dim=1)

    _loss = F.relu(positive.data[0] - negative.data[0] + 0.3).mean()
    p=positive.data[0].mean()
    return _loss, positive.data[0].mean(), negative.data[0].mean()


def scp_loss(scores,feats,targets, criterion_cls, criterion_feature, P, K):
    feats_v, feats_t= feats
    _, features_v, std_features_v = feats_v
    _, features_t, std_features_t = feats_t
    # cls
    loss_cls1 = criterion_cls(scores[0], targets)
    loss_cls2 = criterion_cls(scores[1], targets)
    loss_cls3 = criterion_cls(scores[2], targets)
    loss_cls4 = criterion_cls(scores[3], targets)
    loss_cls = loss_cls1 + loss_cls2 + loss_cls3 + loss_cls4
    # tri
    loss_tri_vt_1, pos_vt_1, neg_vt_1 = triplet_hard_loss(features_v[0],features_t[0], P, K)
    loss_tri_vt_2, pos_vt_2, neg_vt_2 = triplet_hard_loss(features_v[1],features_t[1], P, K)
    loss_tri_vt_3, pos_vt_3, neg_vt_3 = triplet_hard_loss(features_v[2],features_t[2], P, K)
    loss_tri_vt_4, pos_vt_4, neg_vt_4 = triplet_hard_loss(features_v[3],features_t[3], P, K)
    loss_tri_vt = loss_tri_vt_1 + loss_tri_vt_2 + loss_tri_vt_3 + loss_tri_vt_4
    pos_vt = pos_vt_1 + pos_vt_2 + pos_vt_3 + pos_vt_4
    neg_vt = neg_vt_1 + neg_vt_2 + neg_vt_3 + neg_vt_4
    #
    loss_tri_tv_1, pos_tv_1, neg_tv_1 = triplet_hard_loss(features_t[0], features_v[0], P, K)
    loss_tri_tv_2, pos_tv_2, neg_tv_2 = triplet_hard_loss(features_t[1], features_v[1], P, K)
    loss_tri_tv_3, pos_tv_3, neg_tv_3 = triplet_hard_loss(features_t[2], features_v[2], P, K)
    loss_tri_tv_4, pos_tv_4, neg_tv_4 = triplet_hard_loss(features_t[3], features_v[3], P, K)
    loss_tri_tv = loss_tri_vt_1 + loss_tri_vt_2 + loss_tri_vt_3 + loss_tri_vt_4
    pos_tv = pos_tv_1 + pos_tv_2 + pos_tv_3 + pos_tv_4
    neg_tv = neg_tv_1 + neg_tv_2 + neg_tv_3 + neg_tv_4

    # feat
    loss_feat1_v = criterion_feature(features_v[0], std_features_v[0])
    loss_feat2_v = criterion_feature(features_v[1], std_features_v[1])
    loss_feat3_v = criterion_feature(features_v[2], std_features_v[2])
    loss_feat4_v = criterion_feature(features_v[3], std_features_v[3])
    loss_feat_v = loss_feat1_v + loss_feat2_v + loss_feat3_v + loss_feat4_v
    #
    loss_feat1_t = criterion_feature(features_t[0], std_features_t[0])
    loss_feat2_t = criterion_feature(features_t[1], std_features_t[1])
    loss_feat3_t = criterion_feature(features_t[2], std_features_t[2])
    loss_feat4_t = criterion_feature(features_t[3], std_features_t[3])
    loss_feat_t = loss_feat1_t + loss_feat2_t + loss_feat3_t + loss_feat4_t

    loss = loss_cls#+ loss_feat_v * 10 +loss_feat_t *10 #+ loss_tri_vt + loss_tri_tv

    # acc

    _, idx = sum(scores).max(dim=1)
    acc = (idx == targets).float().mean().item()
    return loss, acc

def id_inter_loss(outputs,v_global_feat,t_global_feat,labels):
    id_criterion= torch.nn.CrossEntropyLoss()
    id_loss=0
    batch_size = v_global_feat.size(0)
    tri_loss = TripletLoss_id(margin=0.3)
    label_v=torch.split(labels,batch_size)[0]
    label_t=torch.split(labels,batch_size)[0]

    for i in range(6):
        id_loss+=id_criterion(outputs[i],labels)
    inter_loss_v=global_loss(tri_loss, v_global_feat,t_global_feat,label_v,label_t,normalize_feature=True)
    inter_loss_t = global_loss(tri_loss, t_global_feat, v_global_feat, label_t,label_v, normalize_feature=True)

    inter_loss=inter_loss_v[0]+inter_loss_t[0]

    loss=id_loss+inter_loss#+

    return loss
##new loss (with OIM(instead of id loss) , inter modality(triplet), center loss)
def oim_inter_center_loss(center_criterion,outputs,feat,feat_p,v_global_feat,t_global_feat,labels,num_part,summary_writer,global_step):
    id_criterion= torch.nn.CrossEntropyLoss()
    id_loss=id_criterion(outputs[0],labels)
    # center_loss =0
    n_class=outputs[0].size(1)
    batch_size = v_global_feat.size(0)
    feat_dim=v_global_feat.size(1)
    tri_loss = TripletLoss_id(margin=0.3)
    label_v=torch.split(labels,batch_size)[0]
    label_t=torch.split(labels,batch_size)[1]
    # center_criterion = CenterLoss(num_classes=n_class, feat_dim=feat_dim, use_gpu=True)
    output_v = torch.split(outputs[0], batch_size)[0]
    output_t = torch.split(outputs[0], batch_size)[1]
    id_loss_v = id_criterion(output_v, label_v)
    id_loss_t = id_criterion(output_t, label_t)

    predict_glo=torch.cat((v_global_feat,t_global_feat),0)
    id_glo=id_criterion(predict_glo,labels)
    # center_loss = center_criterion(labels,feat[0])
    tmp_glo=global_loss(tri_loss, feat_p[0], feat_p[0], labels, labels, normalize_feature=True)
    inter_glo = tmp_glo[0]
    out_feat_v=torch.split(feat_p[0], batch_size)[0]
    out_feat_t = torch.split(feat_p[0], batch_size)[1]
    tmp_v=global_loss(tri_loss, out_feat_v,out_feat_t, label_v,label_t, normalize_feature=True)
    tri_loss_v=tmp_v[0]
    tmp_t = global_loss(tri_loss, out_feat_v, out_feat_t, label_v, label_t, normalize_feature=True)
    tri_loss_t=tmp_t[0]

    intra_loss_v_an = torch.split(tmp_glo[4], batch_size)[0]
    intra_loss_t_an = torch.split(tmp_glo[4], batch_size)[1]
    inter_loss_v_ap = tmp_v[3]
    inter_loss_t_ap = tmp_t[3]
    apn_intra = torch.max(intra_loss_t_an, intra_loss_v_an)
    apn_inter = torch.min(inter_loss_v_ap, inter_loss_t_ap)
    inter_intra_pn = tri_loss(apn_intra, apn_inter)

    prec=tmp_glo[-1]+tmp_t[-1]+tmp_v[-1]+(apn_intra.data > apn_inter.data).sum() * 1. / label_v.size(0)
    for i in range(1,num_part):
        id_loss=id_loss+id_criterion(outputs[i],labels)
        output_v = torch.split(outputs[i], batch_size)[0]
        output_t = torch.split(outputs[i], batch_size)[1]
        id_loss_v=id_loss_v+id_criterion(output_v,label_v)
        id_loss_t= id_loss_t +id_criterion(output_t, label_t)
        # center_loss=center_loss+center_criterion(labels,feat[i])
        tmp_glo=global_loss(tri_loss,feat_p[i], feat_p[i],labels,labels, normalize_feature=True)
        inter_glo = inter_glo+tmp_glo[0]

        out_feat_v = torch.split(feat_p[i], batch_size)[0]
        out_feat_t = torch.split(feat_p[i], batch_size)[1]
        tmp_v=global_loss(tri_loss, out_feat_v, out_feat_t, label_v, label_t, normalize_feature=True)
        tmp_t=global_loss(tri_loss, out_feat_t, out_feat_v, label_t, label_v, normalize_feature=True)
        tri_loss_v = tri_loss_v+tmp_v[0]
        tri_loss_t = tri_loss_t+tmp_t[0]

        intra_loss_v_an=torch.split(tmp_glo[4],batch_size)[0]
        intra_loss_t_an = torch.split(tmp_glo[4], batch_size)[1]
        inter_loss_v_ap=tmp_v[3]
        inter_loss_t_ap=tmp_t[3]
        apn_intra = torch.max(intra_loss_t_an, intra_loss_v_an)
        apn_inter = torch.min(inter_loss_v_ap, inter_loss_t_ap)
        inter_intra_pn = tri_loss(apn_intra, apn_inter)

        prec = prec + tmp_glo[-1] + tmp_t[-1] + tmp_v[-1]+(apn_inter.data > apn_intra.data).sum() * 1. / label_v.size(0)

    id_total_loss=id_loss+id_loss_v+id_loss_t


    modal_loss=inter_glo+inter_intra_pn+tri_loss_v+tri_loss_t
    ####center loss in V and T

    #
    # center_loss_v=center_criterion(v_global_feat, label_v)
    # center_loss_t=center_criterion(t_global_feat, label_t)
    #
    # center_loss=center_loss_v+center_loss_t
    # center_loss = center_criterion(labels,feat)
    ###
    loss =1*(0*id_loss+0*id_glo) +1*modal_loss#+0*center_loss#0.5*inter_loss##inter_loss#0.1*center_loss##0.05*center_loss#+ #
    # ######  center loss important for backward training , inter loss play a weak role in loss reduction(the bigger , more impact on the ID loss precision)
    prec=prec*1.0/num_part
    summary_writer.add_scalar('modal_loss', modal_loss.item(), global_step)
    summary_writer.add_scalar('id_loss', id_loss.item(), global_step)
    summary_writer.add_scalar('total_loss', loss.item(), global_step)
    return loss,prec

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


