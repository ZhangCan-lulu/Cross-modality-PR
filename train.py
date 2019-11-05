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
from data_loader import SYSUData, RegDBData, TestData,SYSUData_tri
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb,accuracy
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

def main():
    parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
    parser.add_argument('--arch', default='resnet50', type=str,
                        help='network baseline:resnet18 or resnet50 or se_resnet50 or pcb_rpp or cbam or pcb_pyramid')
    parser.add_argument('--resume', '-r', default='', type=str,
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
    parser.add_argument('--test-batch', default=64, type=int,
                        metavar='tb', help='testing batch size')
    parser.add_argument('--method', default='id', type=str,
                        metavar='m', help='method type:id or triplet or sphere or id_triplet or id_triplet_center or id_inter_loss')
    parser.add_argument('--drop', default=0.0, type=float,
                        metavar='drop', help='dropout ratio')
    parser.add_argument('--trial', default=1, type=int,
                        metavar='t', help='trial (only for RegDB dataset)')
    parser.add_argument('--gpu', default='0,2', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--mode', default='indoor', type=str, help='all or indoor')
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # args.gpu
    np.random.seed(0)

    dataset = args.dataset
    if dataset == 'sysu':
        data_path = "/home/zhangc/projects/Datasets/SYSU-MM01/"
        log_path = args.log_path + 'sysu_log/'
        test_mode = [1,2]#[1,2]  # thermal to visible
    elif dataset == 'regdb':
        data_path = "/home/zhangc/projects/Datasets/RegDB/"
        log_path = args.log_path + 'regdb_log/'
        test_mode = [2, 1]  # visible to thermal

    suffix_id='Debug_1m1_indoor_Avg_Atte'
    lamda_ap=0
    lamda_cen=1

    checkpoint_path = args.model_path

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    if args.method == 'id':
        suffix = dataset + '_id_epoch10_'+suffix_id+'_debug_pyramid_attention'#wr0_serial_train_newepoch
    elif args.method == 'sphere':
        suffix = dataset + '_sphere'
    elif args.method == 'triplet':
        suffix = dataset + '_triplet_debug_sampler2_epoch20'
    elif args.method == 'id_triplet':
        suffix = dataset + '_id_triplet_wr'
    elif args.method == 'id_triplet_center':
        suffix = dataset + '_id_triplet_center'
    elif args.method == 'id_inter_loss':
        suffix = dataset +'_inter_loss_id_apdist_1_1_gloid_pyramid_attention_sampler2'#debug_cbamp_debug_cbamp_
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
        suffix=suffix+'_'+'use_rpp'

    if args.with_se:
        suffix = suffix + '_' + 'use_senet'

    if args.use_cbam:
        suffix = suffix + '_' + 'use_cbam'

    if dataset == 'regdb':
        suffix = suffix + '_trial_{}'.format(args.trial)

    test_log_file = open(log_path + suffix + '.txt', "w")
    sys.stdout = Logger(log_path + suffix + '_os.txt')
    summary_writer = SummaryWriter(os.path.join(log_path, 'tensorboard_log_'+suffix_id+'_debug_pyramid_attention'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0
    feature_dim = args.low_dim

    if args.arch=='pcb_pyramid':
        num_part=10
        suffix = suffix + '_' + 'pcb_pyramid'
    # elif args.arch=='pcb_cro':
    #     num_part=11
    else:
        num_part=6#pcb_rpp method
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

    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

    # testing data loader
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    n_class = len(np.unique(trainset.train_color_label))
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
            triplet_criterion = TripletLoss(margin=0.7)
            triplet_criterion.to(device)
            center_criterion = CenterLoss(num_classes=n_class, feat_dim=args.low_dim, size_average=True).to(device)
            center_criterion_cro = CenterLoss_cro(num_classes=n_class, feat_dim=args.low_dim, size_average=True).to(device)
        criterion.to(device)

    elif args.method == 'id_inter_loss':
        #oim_loss = OIMLoss(args.low_dim, n_class, 1, 0.5).cuda()
        center_criterion = CenterLoss(num_classes=n_class, feat_dim=args.low_dim, size_average=True).to(device)
###########################
    ignored_params = list(map(id, net.feature.parameters())) + list(map(id, net.classifier.parameters()))#+ list(map(id, net.pcb_classifier.parameters()))\
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
        ignored_params = list(map(id, net.classifier.parameters())) #+list(map(id, net.feature.parameters())) #+ list(map(id, net.attention.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        if args.arch == 'pcb_rpp'or 'pcb_pyramid':
            optimizer_pcb = optim.SGD([
                {'params': base_params, 'lr': args.lr},
                # {'params': net.feature.parameters(), 'lr': args.lr},
                {'params': net.classifier.parameters(), 'lr': args.lr * 10}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        return optimizer_pcb

    def pcb_cbam_train(net):
        ignored_params = list(map(id, net.classifier.parameters())) + list(map(id, net.attention.parameters()))+ list(map(id, net.glo_classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        # if args.arch == 'pcb_rpp'or 'pcb_pyramid':
        optimizer_pcb_cbam = optim.SGD([
            {'params': base_params, 'lr': args.lr},
            {'params': net.attention.parameters(), 'lr': args.lr*10},
            {'params': net.glo_classifier.parameters(), 'lr': args.lr*10},
            {'params': net.classifier.parameters(), 'lr': args.lr*10}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        return optimizer_pcb_cbam

    def id_cbam_train(net):
        ignored_params = list(map(id, net.classifier.parameters())) + list(map(id, net.attention.parameters()))+ list(map(id, net.feature.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        # if args.arch == 'pcb_rpp'or 'pcb_pyramid':
        optimizer_id_cbam = optim.SGD([
            {'params': base_params, 'lr': args.lr},
            {'params': net.attention.parameters(), 'lr': args.lr*10},
            {'params': net.feature.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr},
            {'params': center_criterion.parameters(), 'lr': args.lr * 10},
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        return optimizer_id_cbam

    def inter_loss_train(net):
        ignored_params = list(map(id, net.classifier.parameters()))  + list(map(id, net.glo_classifier.parameters()))#+list(map(id, net.feature.parameters()))+list(map(id, net.glo_feature.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        if args.method == 'id_inter_loss':
            optimizer_inter = optim.SGD([
                {'params': base_params, 'lr': args.lr},
                # {'params': net.feature.parameters(), 'lr': args.lr},
                {'params': net.glo_classifier.parameters(), 'lr': args.lr*10},
                # {'params': center_criterion.parameters(), 'lr': args.lr*10},
                {'params': net.classifier.parameters(), 'lr': args.lr * 10}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        return optimizer_inter



    def full_train(net):

        ignored_params = list(map(id, net.feature.parameters()))  + list(map(id, net.pcb_classifier.parameters()))\
            + list(map(id, net.avgpool.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
        optimizer_full = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.feature.parameters(), 'lr': args.lr},
            {'params': net.pcb_classifier.parameters(), 'lr': args.lr},
            {'params': net.avgpool.parameters(), 'lr': args.lr}
        ],weight_decay=5e-4, momentum=0.9, nesterov=True)
        return optimizer_full

    def rpp_train(net):
        optimizer_rpp = optim.SGD([
                {'params': net.avgpool.parameters(), 'lr': args.lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)

        return optimizer_rpp

    '''
    def lr_scheduler(optimizer, epoch):  # new2
        warmup_epoch = 30
        warmup_lr = 1e-5
        lr_steps = [60, 90, 130]
        start_lr = 1e-2
        lr_factor = 0.1

        if epoch <= warmup_epoch:  # lr warmup
            warmup_scale = (start_lr / warmup_lr) ** (1.0 / warmup_epoch)
            lr = warmup_lr * (warmup_scale ** epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.defaults['lr'] = lr
        else:  # lr jump
            for i, el in enumerate(lr_steps):
                if epoch == el:
                    lr = start_lr * (lr_factor ** (i + 1))
                    # logger.info('====> LR is set to: {}'.format(lr))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    optimizer.defaults['lr'] = lr
        lrs = [round(el['lr'], 6) for el in optimizer.param_groups]
        return optimizer, lrs

    
    def warmup_fn(optimizer, epoch):#new1
        if epoch<=30:
            lr=0.1*args.lr*epoch
        elif epoch>30 and epoch <=60:
            lr=args.lr
        elif epoch>60 and epoch<=90:
            lr=args.lr*0.1
        elif epoch>90 and epoch <=120:
            lr=args.lr*0.01

        optimizer.param_groups[0]['lr'] =lr
        optimizer.param_groups[1]['lr'] =lr
        optimizer.param_groups[2]['lr'] =lr

        return lr

    '''
    def adjust_learning_rate(optimizer, epoch,flag):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if epoch <= 10:#30:
            lr = args.lr
        elif epoch > 10 and epoch <=20:
            lr = args.lr * 0.1
        elif epoch > 20 and epoch < 30:
            lr = args.lr * 0.01
        else:
            lr = args.lr * 0.001

        if flag=='pcb':#args.arch=='pcb_rpp'
            optimizer.param_groups[0]['lr'] = lr
            # optimizer.param_groups[1]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10

        elif flag=='rpp':#args.with_rpp
            optimizer.param_groups[0]['lr'] = lr
        elif flag=='full':#full_train
            optimizer.param_groups[0]['lr'] = 0.1 * lr
            optimizer.param_groups[1]['lr'] = lr
            optimizer.param_groups[2]['lr'] = lr
            optimizer.param_groups[3]['lr'] = lr
        elif flag=='normal':
            optimizer.param_groups[0]['lr'] = 0.1*lr
            optimizer.param_groups[1]['lr'] = lr
            optimizer.param_groups[2]['lr'] = lr
        elif flag == 'id_inter_loss':
            optimizer.param_groups[0]['lr'] = lr
            # optimizer.param_groups[1]['lr'] = lr
            # optimizer.param_groups[2]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10
            optimizer.param_groups[2]['lr'] = lr * 10
        elif flag=='pcb_cbam':
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr*10
            optimizer.param_groups[2]['lr'] = lr * 10
            optimizer.param_groups[3]['lr'] = lr*10

        elif flag=='id_cbam':
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr*10
            optimizer.param_groups[2]['lr'] = lr
            optimizer.param_groups[3]['lr'] = lr
            optimizer.param_groups[4]['lr'] = lr * 10

        return lr

    def adjust_lr(optimizer, ep,flag='normal'):
        if ep < 40:
            lr = 1e-3 * (ep // 5 + 1)
        elif ep < 60:
            lr = 1e-2
        elif ep < 80:
            lr = 1e-3
        else:
            lr = 1e-4
        for p in optimizer.param_groups:
            p['lr'] = lr

        return lr
#################################################################################################



    def train(net,epoch,optimizer,flag,ini_id,ini_modal):
        current_lr = adjust_learning_rate(optimizer, epoch,flag)
        # current_lr = warmup_fn(optimizer, epoch)
       # _, current_lr = lr_scheduler(optimizer, epoch)
        train_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        precisions = AverageMeter()
        correct = 0
        total = 0

        print('==> Preparing Data Loader...')
        # identity sampler
        # sampler = IdentitySampler(trainset.train_color_label, \
        #                           trainset.train_thermal_label, color_pos, thermal_pos, args.batch_size)
        sampler = RandomIdentitySampler_alignedreid(trainset.train_color_label, \
                                  trainset.train_thermal_label, color_pos, thermal_pos, args.batch_size,4)

        # sampler = triplet_Sampler_reid(trainset.train_color_label, \
        #                           trainset.train_thermal_label, color_pos, thermal_pos, args.batch_size,8)
        trainset.cIndex = sampler.index1  # color index
        trainset.tIndex = sampler.index2  # thermal index
        # trainset.anchor_pos_rgb = sampler.anchor_pos_rgb
        # trainset.anchor_pos_ir = sampler.anchor_pos_ir
        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, \
                                      sampler=sampler, num_workers=args.workers, drop_last=True)


        # trainset=SYSU_triplet_dataset(data_folder=data_path)
        # trainloader=data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,drop_last = True)
        # switch to train mode
        net.train()
        model_static=net.visible_net.visible.state_dict()
        end = time.time()

        for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
        # for batch_idx, (anchor_r,anchor_t,input1, input2, label1, label2,an_label1,an_label2) in enumerate(trainloader):
            ####show attention map
            # img1=input1
            # img2=input2
            # show_img=img1[-1,:,:,:]
            # plt.imshow(show_img)
            # plt.figure()
            # target_index = None
            # mask_v = net.grad_cam_v(img1, target_index)
            # mask_t = net.grad_cam_t(img2, target_index)
            # show_cam_on_image(img1[-1,:,:,:], mask_v,epoch,mode="visible")
            # show_cam_on_image(img2[-1,:,:,:], mask_t, epoch,mode="thermal")

            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())

            label1=Variable(label1.cuda())
            label2=Variable(label2.cuda())

            labels = torch.cat((label1, label2), 0)
            labels = Variable(labels.cuda())
            data_time.update(time.time() - end)

            output, feat = net(input1, input2)


            global_step = epoch * len(trainloader) + batch_idx

            summary_writer.add_scalar('lr', current_lr, global_step)

            if args.method == 'id':

                if args.arch == 'pcb_rpp' or args.arch =='pcb_pyramid':
                    outputs, v_global_feat, t_global_feat, feat_p = output
                    loss = criterion(outputs[0], labels)
                    prec, = accuracy(outputs[0], labels.data, topk=(20,))
                    correct_lc = (outputs[0].max(1)[1]).eq(labels).sum().item()
                    for i in range(1, num_part):
                        loss += criterion(outputs[i], labels)
                        tmp_prec, = accuracy(outputs[i], labels.data, topk=(20,))
                        prec += tmp_prec
                        # prec1 = torch.mean(prec[0])
                        correct_lc += (outputs[i].max(1)[1]).eq(labels).sum().item()
                    prec1 = prec[0] * 1.0 / num_part
                    correct += correct_lc * 1.0 / num_part
                else:
                    # net.classifier.to(device)
                    # output=net.classifier(output,labels)

                    id_loss = criterion(output, labels)
                    glo_modal,_,_,_=triplet_criterion(feat, feat, labels)
                    feat_glo_v=torch.split(feat,args.batch_size,0)[0]
                    feat_glo_t = torch.split(feat, args.batch_size, 0)[1]
                    modal_glo_v,_ ,dist_ap_v,dist_an_v= triplet_criterion(feat_glo_v, feat_glo_v, label1)
                    modal_glo_t,_,dist_ap_t,dist_an_t = triplet_criterion(feat_glo_t, feat_glo_t, label2)

                    intra_modal=modal_glo_v+modal_glo_t
                    inter_modal_glo_v, _, inter_dist_ap_v, inter_dist_an_v = triplet_criterion(feat_glo_v, feat_glo_t, label1)
                    inter_modal_glo_t, _, inter_dist_ap_t, inter_dist_an_t = triplet_criterion(feat_glo_t, feat_glo_v, label1)
                    inter_modal=inter_modal_glo_v+inter_modal_glo_t
                    # ap_dist_loss=torch.mean(inter_dist_ap_v)+torch.mean(inter_dist_ap_t)+torch.mean(dist_ap_v)+torch.mean(dist_ap_t)

                    apn_intra = torch.max(dist_an_v, dist_an_t)
                    apn_inter = torch.min(inter_dist_ap_v, inter_dist_ap_t)
                    inter_intra_pn = triplet_criterion.dist_l2_apn(feat_glo_v, feat_glo_t)

                    modal_loss=glo_modal+0*intra_modal+0*inter_modal+0*inter_intra_pn

                    center_loss=center_criterion(labels,feat)
                    # center_loss = center_criterion_cro(label1, label2,feat_glo_v,feat_glo_t)
                    delta_id=np.maximum(1.0*(ini_id-id_loss.cpu().detach().numpy())/(id_loss.cpu().detach().numpy()+1e-12),0)
                    delta_modal=np.maximum(1.0*(ini_modal-modal_loss.cpu().detach().numpy())/(modal_loss.cpu().detach().numpy()+1e-12),0)

                    if not delta_id and not delta_modal:
                        alpha=1
                    else:
                        alpha=delta_id*1.0/(delta_id+delta_modal)

                    loss=1*id_loss+lamda_ap*modal_loss+lamda_cen*center_loss
                    summary_writer.add_scalar('modal_loss', modal_loss.item(), global_step)
                    summary_writer.add_scalar('id_loss', id_loss.item(), global_step)
                    summary_writer.add_scalar('center_loss', center_loss.item(), global_step)
                    summary_writer.add_scalar('total_loss', loss.item(), global_step)
                    prec, = accuracy(output.data, labels.data, topk=(20,))
                    prec1 = prec[0]
                    correct += (output.max(1)[1]).eq(labels).sum().item()
                    ini_id = id_loss.cpu().detach().numpy()
                    ini_modal = modal_loss.cpu().detach().numpy()
                # _, predicted = outputs.max(1)
                # correct += predicted.eq(labels).sum().item()
                # revise by zc

            elif args.method == 'id_inter_loss':
                #oim_loss = OIMLoss(n_class, n_class, 30, 0.5).cuda()
                # loss = id_inter_loss(outputs,v_global_feat,t_global_feat,labels)
                outputs,v_global_feat,t_global_feat,feat_p=output
                loss,prec = oim_inter_center_loss(center_criterion,outputs,feat, feat_p,v_global_feat, t_global_feat, labels,num_part,summary_writer,global_step)

                # prec1=prec
                # correct_lc=prec*6
                prec, = accuracy(outputs[0], labels.data, topk=(20,))
                correct_lc = (outputs[0].max(1)[1]).eq(labels).sum().item()
                for i in range(1, num_part):
                    tmp_prec, = accuracy(outputs[i], labels.data, topk=(20,))
                    prec += tmp_prec
                    # prec1 = torch.mean(prec[0])
                    correct_lc += (outputs[i].max(1)[1]).eq(labels).sum().item()
                prec1 = prec[0]*1.0 / num_part
                correct += correct_lc * 1.0 / num_part

            elif args.method == 'sphere':

                loss = sphere_criterion(feat, labels)
                prec, = accuracy(output.data, labels.data, topk=(20,))
                prec1 = prec[0]

            elif args.method == 'triplet':
                if args.arch=="pcb_rpp"or  args.arch=='pcb_pyramid':
                     loss,prec= triplet_criterion(feat_p[0], labels)
                     correct_lc = (feat_p[0].max(1)[1]).eq(labels).sum().item()
                     for i in range(1,num_part):
                         loss_tmp,prec_tmp =triplet_criterion(feat_p[i],labels)
                         loss+=loss_tmp
                         prec+=prec_tmp
                #
                         correct_lc += (outputs[i].max(1)[1]).eq(labels).sum().item()
                     prec1 = prec*1.0 / num_part
                     correct += correct_lc * 1.0 / num_part
                else:

                    loss, prec1 = triplet_criterion(feat, labels)
                    correct+=prec1


            elif args.method == 'id_triplet':
                loss = 0.3 * id_criterion(outputs, labels) + 0.7 * triplet_criterion(feat, labels)[0]
                prec, = accuracy(outputs.data, labels.data, topk=(20,))
                prec1 = prec[0]

            elif args.method == 'id_triplet_center':
                loss = id_criterion(outputs, labels) + triplet_criterion(feat, labels)[
                    0] + Center_weight_loss * center_criterion(feat, labels)
                prec, = accuracy(outputs.data, labels.data, topk=(20,))
                prec1 = prec[0]




            total += labels.size(0)
            # acc_avg=(outputs.max(1)[1]==labels).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), labels.size(0))  # loss.item()
            precisions.update(prec1, labels.size(0))
            # total += labels.size(0)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 10 == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time: {:.3f} ({:.3f})\t'
                      'Data: {:.3f} ({:.3f})\t'
                      'lr:{}\t'
                      'Loss: {:.4f} ({:.4f})\t'
                      'rank1 {:.2%} ({:.2%})\t'
                      'acc {:.2%}\t '
                      .format(epoch, batch_idx, len(trainloader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              current_lr,
                              train_loss.val, train_loss.avg,
                              precisions.val, precisions.avg,
                              1. * correct / total))

            # if epoch % 2 == 0 and batch_idx==len(trainloader):
            #     figure=draw_cluster(net.l2norm(output),labels,32,32)
            #
            #     plt.savefig('/home/zhangc/projects/cross_modality_V/BDTR_modal_loss_cbam_debug/image/cluster/'+str(epoch)+'_train_'+suffix_id+'.jpg')
            #     plt.show()
            #     plt.pause(1)
            #     plt.close()
##############################################################################################

    def test(net,epoch):
        # switch to evaluation mode
        net.eval()
        print('Extracting Gallery Feature...')
        start = time.time()
        ptr = 0
        gall_feat = np.zeros((ngall, args.low_dim))#args.low_dim
        label_g = []
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat_pool, feat = net(input, input, test_mode[0])
                label_g.append(label)
                gall_feat[ptr:ptr + batch_num,:] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))

        # switch to evaluation mode
        net.eval()
        print('Extracting Query Feature...')
        start = time.time()
        ptr = 0
        query_feat = np.zeros((nquery,args.low_dim))
        label_q=[]
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(query_loader):
                batch_num = input.size(0)
                input = Variable(input.cuda())
                feat_pool, feat = net(input, input, test_mode[1])

                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                label_q.append(label)
                ptr = ptr + batch_num
        print('Extracting Time:\t {:.3f}'.format(time.time() - start))

        start = time.time()
        # compute the similarity
        distmat = np.matmul(query_feat, np.transpose(gall_feat))

        # output=np.concatenate((query_feat,gall_feat),0)
        # labels=label_q+label_g
        # figure = draw_cluster(output, labels, len(label_q),len(label_g))
        # #
        # plt.savefig('/home/zhangc/projects/cross_modality_V/BDTR_modal_loss_cbam_debug/image/cluster/' + str(
        #     epoch) + '_test_'+suffix_id+'.jpg')
        # plt.show()
        # plt.pause(1)
        # plt.close()

        # evaluation
        if dataset == 'regdb':
            cmc, mAP = eval_regdb(-distmat, query_label, gall_label)
        elif dataset == 'sysu':
            cmc, mAP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        print('Evaluation Time:\t {:.3f}'.format(time.time() - start))



        # uni_que = np.unique(query_label)
        # n_q, n_g = 0, 0
        # output = []
        # labels = []
        # for i in range(30):
        #     label_id = uni_que[randrange(0, len(uni_que))]
        #     index_q = [i for i, a in enumerate(query_label) if a == label_id]
        #     index_g = [i for i, a in enumerate(gall_label) if a == label_id]
        #     output.extend(np.concatenate((query_feat[index_q, :], gall_feat[index_g, :]), 0))
        #     tmp_q = query_label[index_q]
        #     tmp_g = gall_label[index_g]
        #     n_q += len(tmp_q)
        #     n_g += len(tmp_g)
        #     labels.extend(np.concatenate((tmp_q, tmp_g), 0))
        #
        # figure = draw_cluster(output, labels, n_q, n_g)
        # #
        # suffix_id = 'DEBUG_30'
        # plt.savefig(
        #     '/home/zhangc/projects/cross_modality_V/BDTR_modal_loss_cbam_debug/image/cluster/'  + '_train_' + suffix_id + '.jpg')
        # plt.show()
        # plt.pause(1)
        # plt.close()
        return cmc, mAP
#######################################################################################################
        # training


    def train_model(net,optimizer,final_epoch,flag='normal'):
        # training
        print('==> Start Training...')
        best_acc = 0
        for epoch in range(start_epoch, final_epoch - start_epoch+1):


            train(net, epoch, optimizer, flag,ini_id,ini_modal)


            if epoch % 2 == 0:#epoch > 0 and
                print('Test Epoch: {}'.format(epoch))
                print('Test Epoch: {}'.format(epoch), file=test_log_file)
                # testing
                cmc, mAP = test(net, epoch)
                print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP))
                print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP), file=test_log_file)
                test_log_file.flush()

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
#######################################################################################################

    if args.arch == 'pcb_rpp'or args.arch == 'pcb_pyramid':
        # print('-------epoch for pcb_traing--------')
        # optimizer_pcb = pcb_train(net)
        # model=train_model(net, optimizer_pcb, 60, flag='pcb')

        if args.with_rpp:
            print('-------epoch for rpp_traing--------')
            model = model.convert_to_rpp().cuda()
            optimizer_rpp = rpp_train(model)
            model=train_model(model, optimizer_rpp, 40, flag='rpp')

            print('-------epoch for full_traing--------')
            optimizer_full = full_train(model)
            train_model(model, optimizer_full, 80, flag='full')
        elif args.method=='id_inter_loss':

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


if __name__=='__main__':

    main()