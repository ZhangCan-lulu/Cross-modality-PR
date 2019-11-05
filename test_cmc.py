from __future__ import print_function
import argparse
import logging
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData, SYSUData_tri,TestData_test
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, accuracy ,eval_sysu_test
from model import embed_net
from utils import *
from loss_fn.triplet_loss import *
from loss_fn.center_loss import *
from loss_fn.sphere_loss import *
from loss_fn.scp_loss import *
from loss_fn.OIM_loss import *
from senet.grad_cbam import *
from loss_fn.cauthy_hash import *
from tensorboardX import SummaryWriter
import numpy as np
from draw_custer import *
from sklearn.manifold import TSNE
from random import randrange

import matplotlib.pyplot as plt
# Baseline +glo_id   sysu_id_epoch10_baseline_gloid_debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_best.t
#
# PPAA+glo_id +center  sysu_id_epoch10_debug_center_0.1_nomodal_baseline_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t
#PPAA+glo_id +exp_center   sysu_id_epoch10_debug_gloid_expcenter_pcb_cbam_debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t
####margin
# all-search_best   sysu_id_epoch10_PPAM_gloid_expcen_1m1_debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t
# all-search_worst   sysu_id_epoch10_PPAM_gloid_expcenter_1m050__debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t
#   sysu_id_epoch10_PPAM_gloid_expcenter_1m0000__debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t
# indoor-search _best sysu_id_epoch10_PPAM_gloid_expcen_1m1_indoor__debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t
# indoor-search_worst  sysu_id_epoch10_PPAM_gloid_expcen_1m2_indoor__debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t
#   sysu_id_epoch10_PPAM_gloid_expcen_1m0_indoor__debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t


###weight
#all_search  sysu_id_epoch10_PPAM_gloid_expcen_1m1_debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t
##all-search_worst  sysu_id_epoch10_PPAM_gloid_expcen_3m1_all__debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t
#  sysu_id_epoch10_PPAM_gloid_expcen_0m1_all__debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t

# indoor-search _best sysu_id_epoch10_PPAM_gloid_expcen_1m1_indoor__debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t
# indoor-search_worst   sysu_id_epoch10_PPAM_gloid_expcen_3m1_indoor__debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t
# sysu_id_epoch10_PPAM_gloid_expcen_0m0_indoor__debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t

def main():
    parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
    parser.add_argument('--arch', default='resnet50', type=str,
                        help='network baseline:resnet18 or resnet50 or se_resnet50 or pcb_rpp or cbam or pcb_pyramid')
    parser.add_argument('--resume', '-r',
                        default='sysu_id_epoch10_PPAM_gloid_expcenter_1m0000__debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t',
                        help='resume from checkpoint')
    parser.add_argument('--test-only', action='store_true', help='test only')
    parser.add_argument('--model_path', default='save_model/', type=str,
                        help='model save path')
    parser.add_argument('--save_epoch', default=20, type=int,
                        metavar='s', help='save model every 10 epochs')
    parser.add_argument('--log_path', default='log/', type=str,
                        help='log save path')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--low-dim', default=512, type=int,
                        metavar='D', help='feature dimension')
    parser.add_argument('--img_w', default=144, type=int,
                        metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int,
                        metavar='imgh', help='img height')
    parser.add_argument('--batch-size', default=32, type=int,
                        metavar='B', help='training batch size')
    parser.add_argument('--test-batch', default=1, type=int,
                        metavar='tb', help='testing batch size')
    parser.add_argument('--method', default='id', type=str,
                        metavar='m',
                        help='method type:id or triplet or sphere or id_triplet or id_triplet_center or id_inter_loss')
    parser.add_argument('--drop', default=0.0, type=float,
                        metavar='drop', help='dropout ratio')
    parser.add_argument('--trial', default=1, type=int,
                        metavar='t', help='trial (only for RegDB dataset)')
    parser.add_argument('--gpu', default='0,2', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--mode', default='all', type=str, help='all or indoor')
    ##add by zc
    parser.add_argument('--with_se', default=False, type=bool,
                        help='whether add SEModule to the last conv layer')
    parser.add_argument('--with_rpp', default=False, type=bool,
                        help='whether add rpp with PCB to the last conv layer')
    parser.add_argument('--use_cbam', default=True, type=bool,
                        help='whether add CBAM to the last conv layer')
    parser.add_argument('--reduction', default=16, type=int,
                        help='SEModule reduction ratio')
    parser.add_argument('--with_labelsmooth', default=False, type=bool,
                        help='whether add label smooth to loss function')
    parser.add_argument('--with_model_neck', default='no', type=str,
                        help='whether add bnneck to loss function')
    # Balanced weight of center loss
    Center_weight_loss = 0.01
    ##end with zc
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # args.gpu
    np.random.seed(0)

    dataset = args.dataset
    if dataset == 'sysu':
        data_path = "/home/zhangc/projects/Datasets/SYSU-MM01/"#"/home/zhangc/projects/Datasets/test/"
        log_path = args.log_path + 'sysu_log/'
        test_mode = [1, 2]  # thermal to visible
    elif dataset == 'regdb':
        data_path = "/home/zhangc/projects/Datasets/RegDB/"
        log_path = args.log_path + 'regdb_log/'
        test_mode = [2, 1]  # visible to thermal

    suffix_id = '_DEBUG_'
    lamda_ap = 0
    lamda_cen = 1

    checkpoint_path = args.model_path

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    if args.method == 'id':
        suffix = dataset + '_id_epoch10_' + suffix_id + '_debug_pyramid_attention'  # wr0_serial_train_newepoch
    elif args.method == 'sphere':
        suffix = dataset + '_sphere'
    elif args.method == 'triplet':
        suffix = dataset + '_triplet_debug_sampler2_epoch20'
    elif args.method == 'id_triplet':
        suffix = dataset + '_id_triplet_wr'
    elif args.method == 'id_triplet_center':
        suffix = dataset + '_id_triplet_center'
    elif args.method == 'id_inter_loss':
        suffix = dataset + '_inter_loss_id_apdist_1_1_gloid_pyramid_attention_sampler2'  # debug_cbamp_debug_cbamp_
    if args.with_labelsmooth:
        suffix = suffix + '_ls'

    # suffix = suffix + '_cenloss_{}'.format(Center_weight_loss)
    suffix = suffix + '_drop_{}'.format(args.drop)
    suffix = suffix + '_lr_{:1.1e}'.format(args.lr)
    suffix = suffix + '_dim_{}'.format(args.low_dim)
    if not args.optim == 'sgd':
        suffix = suffix + '_' + args.optim
    suffix = suffix + '_' + args.arch
    if args.with_rpp:
        suffix = suffix + '_' + 'use_rpp'

    if args.with_se:
        suffix = suffix + '_' + 'use_senet'

    if args.use_cbam:
        suffix = suffix + '_' + 'use_cbam'

    if dataset == 'regdb':
        suffix = suffix + '_trial_{}'.format(args.trial)

    test_log_file = open(log_path + '.txt', "w")
    cmc_log_file = open('all' + '_cmc.txt', "w")
    sys.stdout = Logger(log_path + suffix + '_os.txt')
    summary_writer = SummaryWriter(os.path.join(log_path, 'tensorboard_log_' + suffix_id + '_debug_pyramid_attention'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0
    feature_dim = args.low_dim

    if args.arch == 'pcb_pyramid':
        num_part = 10
        suffix = suffix + '_' + 'pcb_pyramid'
    # elif args.arch=='pcb_cro':
    #     num_part=11
    else:
        num_part = 6  # pcb_rpp method
    print('==> Loading data..')
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Pad(10),
        transforms.RandomCrop((args.img_h, args.img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    end = time.time()
    if dataset == 'sysu':
        # training set
        # trainset = SYSUData_tri(data_path, transform=transform_train)
        trainset = SYSUData(data_path, transform=transform_train)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    elif dataset == 'regdb':
        # training set
        trainset = RegDBData(data_path, args.trial, transform=transform_train)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

    gallset = TestData_test(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    queryset = TestData_test(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

    # testing data loader
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    n_class = 395#len(np.unique(trainset.train_color_label))
    nquery = len(query_label)
    ngall = len(gall_label)

    print('Dataset {} statistics:'.format(dataset))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
    print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
    print('  ------------------------------')
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    print('==> Building model..')
    net = embed_net(args.low_dim, n_class, drop=args.drop, arch=args.arch, neck=args.with_model_neck,
                    with_se=args.with_se, with_rpp=args.with_rpp, use_cbam=args.use_cbam, reduction=args.reduction)
    net.to(device)
    cudnn.benchmark = True

    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            # start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
            start_epoch = int(checkpoint['epoch'])
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    if args.method == 'id':
        if args.with_labelsmooth:
            criterion = CrossEntropyLabelSmooth(n_class)
            print("---------using ID loss with label smoothing-------")
        else:
            criterion = nn.CrossEntropyLoss()
            # criterion = FocalLoss(gamma=2)
            print("---------using ID loss only-------")
            triplet_criterion = TripletLoss(margin=0.3)
            triplet_criterion.to(device)
            center_criterion = CenterLoss(num_classes=n_class, feat_dim=args.low_dim, size_average=True).to(device)
            center_criterion_cro = CenterLoss_cro(num_classes=n_class, feat_dim=args.low_dim, size_average=True).to(
                device)
        criterion.to(device)

    elif args.method == 'sphere':
        sphere_criterion = OhemSphereLoss(args.low_dim, n_class)
        sphere_criterion.to(device)
        print("---------using sphere loss -------")

    elif args.method == 'triplet':
        triplet_criterion = TripletLoss(margin=0.3)
        triplet_criterion.to(device)
        print("---------using triplet loss-------")


    ###########################
    ignored_params = list(map(id, net.feature.parameters())) + list(
        map(id, net.classifier.parameters()))  # + list(map(id, net.pcb_classifier.parameters()))\
    # + list(map(id, net.visible_net.visible.avgpool.parameters()))+ list(map(id, net.thermal_net.thermal.avgpool.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    if args.optim == 'sgd':

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.feature.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)

    elif args.optim == 'adam':
        optimizer = optim.Adam([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.feature.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr}], weight_decay=5e-4)

    ####################################################################################################
    def pcb_train(net):
        ignored_params = list(map(id,
                                  net.classifier.parameters()))  # +list(map(id, net.feature.parameters())) #+ list(map(id, net.attention.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        if args.arch == 'pcb_rpp' or 'pcb_pyramid':
            optimizer_pcb = optim.SGD([
                {'params': base_params, 'lr': args.lr},
                # {'params': net.feature.parameters(), 'lr': args.lr},
                {'params': net.classifier.parameters(), 'lr': args.lr * 10}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        return optimizer_pcb

    def pcb_cbam_train(net):
        ignored_params = list(map(id, net.classifier.parameters())) + list(map(id, net.attention.parameters())) + list(
            map(id, net.glo_classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        # if args.arch == 'pcb_rpp'or 'pcb_pyramid':
        optimizer_pcb_cbam = optim.SGD([
            {'params': base_params, 'lr': args.lr},
            {'params': net.attention.parameters(), 'lr': args.lr * 10},
            {'params': net.glo_classifier.parameters(), 'lr': args.lr * 10},
            {'params': net.classifier.parameters(), 'lr': args.lr * 10}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        return optimizer_pcb_cbam

    def id_cbam_train(net):
        ignored_params = list(map(id, net.classifier.parameters())) + list(map(id, net.attention.parameters())) + list(
            map(id, net.feature.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        # if args.arch == 'pcb_rpp'or 'pcb_pyramid':
        optimizer_id_cbam = optim.SGD([
            {'params': base_params, 'lr': args.lr},
            {'params': net.attention.parameters(), 'lr': args.lr * 10},
            {'params': net.feature.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr},
            {'params': center_criterion.parameters(), 'lr': args.lr * 10},
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        return optimizer_id_cbam

    def inter_loss_train(net):
        ignored_params = list(map(id, net.classifier.parameters())) + list(map(id,
                                                                               net.glo_classifier.parameters()))  # +list(map(id, net.feature.parameters()))+list(map(id, net.glo_feature.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        if args.method == 'id_inter_loss':
            optimizer_inter = optim.SGD([
                {'params': base_params, 'lr': args.lr},
                # {'params': net.feature.parameters(), 'lr': args.lr},
                {'params': net.glo_classifier.parameters(), 'lr': args.lr * 10},
                # {'params': center_criterion.parameters(), 'lr': args.lr*10},
                {'params': net.classifier.parameters(), 'lr': args.lr * 10}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        return optimizer_inter

    def full_train(net):

        ignored_params = list(map(id, net.feature.parameters())) + list(map(id, net.pcb_classifier.parameters())) \
                         + list(map(id, net.avgpool.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        optimizer_full = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.feature.parameters(), 'lr': args.lr},
            {'params': net.pcb_classifier.parameters(), 'lr': args.lr},
            {'params': net.avgpool.parameters(), 'lr': args.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        return optimizer_full

    def rpp_train(net):
        optimizer_rpp = optim.SGD([
            {'params': net.avgpool.parameters(), 'lr': args.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        return optimizer_rpp



    #################################################################################################
    def attention_map_show(net,img,feature,classifier,mode):
        grad_cam_v = GradCam(model=net, target_layer_names=["layer4"], use_cuda=True)


        show_img=np.transpose(img.tolist(),[1,2,0])
        plt.imshow(show_img)
        plt.figure()
        target_index = None
        mask_v = grad_cam_v(torch.unsqueeze(img,0), target_index)
        # mask_t = net.grad_cam_t(img2, target_index)
        show_cam_on_image(img, mask_v,mode=mode)



    ##############################################################################################

    def test(net, epoch):
        # switch to evaluation mode
        net.eval()
        print('Extracting Gallery Feature...')
        start = time.time()
        ptr = 0
        gall_feat = np.zeros((ngall, args.low_dim))  # args.low_dim
        gall_file=[]
        label_g = []
        with torch.no_grad():
            for batch_idx, (input, label,img_file) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat_pool, feat = net(input, input, test_mode[0])
                label_g.append(label)
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                gall_file.append(img_file)
                ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))

        # switch to evaluation mode
        net.eval()
        print('Extracting Query Feature...')
        start = time.time()
        ptr = 0
        query_feat = np.zeros((nquery, args.low_dim))
        query_file = []
        with torch.no_grad():
            for batch_idx, (input, label,img_file) in enumerate(query_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())

                feat_pool, feat = net(input, input, test_mode[1])
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                query_file.append(img_file)
                ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))
        img = input[0]
        # attention_map_show(net, img, feat[0], net.classifier, 'query')
        start = time.time()
        # compute the similarity
        distmat = np.matmul(query_feat, np.transpose(gall_feat))

        # evaluation
        if dataset == 'regdb':
            cmc, mAP = eval_regdb(-distmat, query_label, gall_label)
        elif dataset == 'sysu':
            # cmc, mAP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
            cmc, mAP = eval_sysu_test(-distmat, query_label, gall_label, query_cam, gall_cam, np.array(query_img), np.array(gall_img))
        print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

        # uni_que = np.unique(query_label)
        # out_q=[]
        # out_g=[]
        # label_q=[]
        # label_g=[]
        # num_label=20
        # for i in range(num_label):
        #     label_id = uni_que[i]
        #     index_q = [i for i, a in enumerate(query_label) if a == label_id]
        #     index_g = [i for i, a in enumerate(gall_label) if a == label_id]
        #     out_q.extend(query_feat[index_q[:10], :])
        #     out_g.extend(gall_feat[index_g[:10], :])
        #     # tmp_q = query_label[index_q]
        #     # tmp_g = gall_label[index_g]
        #     label_q.extend(query_label[index_q[:10]])
        #     label_g.extend(gall_label[index_g[:10]])
        #     # n_q += len(tmp_q)
        #     # n_g += len(tmp_g)
        #     # labels.extend(np.concatenate((tmp_q, tmp_g), 0))
        # output=np.concatenate((out_q, out_g), 0)
        # labels=np.concatenate((label_q, label_g), 0)
        # n_q = len(out_q)
        # n_g = len(out_g)
        # figure = draw_cluster(output, labels, n_q,n_g,num_label)
        # #
        # suffix_id = 'DEBUG_PPAM+glo_id +expcen_'+str(num_label)
        # plt.savefig(
        #     '/home/zhangc/projects/cross_modality_V/BDTR_modal_loss_cbam_debug/image/cluster/' + '_train_' + suffix_id + '.eps')
        # plt.show()
        # plt.pause(1)
        # plt.close()
        return cmc, mAP

    #######################################################################################################
    # training

    def train_model(net, optimizer, final_epoch, flag='normal'):
        # training
        print('==> Start Training...')
        best_acc = 0
        for epoch in range(start_epoch, final_epoch - start_epoch + 1):

            # train(net, epoch, optimizer, flag,ini_id,ini_modal)

            if epoch % 2 == 0:  # epoch > 0 and
                print('Test Epoch: {}'.format(epoch))
                print('Test Epoch: {}'.format(epoch), file=test_log_file)
                # testing
                cmc, mAP = test(net, epoch)
                print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP))
                print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP), file=test_log_file)
                print(cmc[:50],file=cmc_log_file)
                np.savetxt("./cmc_all_0_mar.txt",cmc,fmt='%f',delimiter=',')
                test_log_file.flush()
                cmc_log_file.flush()
                # save model
                if cmc[0] > best_acc:  # not the real best for sysu-mm01
                    best_acc = cmc[0]
                    state = {
                        'net': net.state_dict(),
                        'cmc': cmc,
                        'mAP': mAP,
                        'epoch': epoch,
                    }
                    torch.save(state, checkpoint_path + suffix + '_best.t')

                # save model every 20 epochs
                if epoch > 10 and epoch % args.save_epoch == 0:
                    state = {
                        'net': net.state_dict(),
                        'cmc': cmc,
                        'mAP': mAP,
                        'epoch': epoch,
                    }
                    torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

            # torch.save(state, checkpoint_path + suffix + '_latest.t')
        return net

    #########################################################################

    ini_id = 10
    ini_modal = 10
    # def train_model(net, optimizer, final_epoch, flag='normal'):
    #     print('==> Start Training...')
    #     best_acc = 0
    #     for epoch in range(start_epoch, final_epoch - start_epoch + 1):
    #         test(net, epoch)
    #     return net
    #######################################################################################################

    if args.arch == 'pcb_rpp' or args.arch == 'pcb_pyramid':
        # print('-------epoch for pcb_traing--------')
        # optimizer_pcb = pcb_train(net)
        # model=train_model(net, optimizer_pcb, 60, flag='pcb')

        if args.with_rpp:
            print('-------epoch for rpp_traing--------')
            model = model.convert_to_rpp().cuda()
            optimizer_rpp = rpp_train(model)
            model = train_model(model, optimizer_rpp, 40, flag='rpp')

            print('-------epoch for full_traing--------')
            optimizer_full = full_train(model)
            train_model(model, optimizer_full, 80, flag='full')
        elif args.method == 'id_inter_loss':

            if args.use_cbam:
                print('-------epoch for pcb_cbam training--------')
                optimizer_pcb_cbam = pcb_cbam_train(net)
                train_model(net, optimizer_pcb_cbam, 90, flag='pcb_cbam')
            else:
                print('-------epoch for id_inter_loss_training--------')
                optimizer_inter = inter_loss_train(net)
                train_model(net, optimizer_inter, 100, flag='id_inter_loss')
            # optimizer_pcb = pcb_train(net)
            # train_model(net, optimizer_pcb, 60, flag='pcb')


        else:

            print('-------epoch for pcb_traing--------')
            optimizer_pcb = pcb_train(net)
            train_model(net, optimizer_pcb, 90, flag='pcb')


    else:
        if args.use_cbam:
            print('-------epoch for resnet_cbam training--------')
            optimizer_id_cbam = id_cbam_train(net)
            train_model(net, optimizer_id_cbam, 90, flag='pcb_cbam')
        else:
            train_model(net, optimizer, 500, flag='normal')


if __name__ == '__main__':
    main()