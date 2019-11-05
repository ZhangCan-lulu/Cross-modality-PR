from __future__ import print_function
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
import time
import numpy as np
import scipy.io as scio
from draw_custer import *
import matplotlib.pyplot as plt
from random import randrange

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50 or se_resnet50 or pcb_rpp or cbam or pcb_pyramid')
parser.add_argument('--resume', '-r', default='sysu_id_epoch10_debug_gloid_expcenter_pcb_cbam_debug_pyramid_attention_drop_0.0_lr_1.0e-02_dim_512_resnet50_use_cbam_best.t', type=str,
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
##end with zc
args = parser.parse_args() 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

np.random.seed(1)
dataset = args.dataset
if dataset == 'sysu':
    data_path = "/home/zhangc/projects/Datasets/SYSU-MM01/"
    log_path = args.log_path + 'sysu_log/'
    test_mode = [2,1]  # thermal to visible
elif dataset == 'regdb':
    data_path = "/home/zhangc/projects/Datasets/RegDB/"
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal

if not os.path.isdir(log_path):
    os.makedirs(log_path)
resume=args.resume
test_log_file = open(log_path + '_test.txt', "w")
sys.stdout = Logger(log_path + '_test_os.txt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
n_class=395
print('==> Building model..')
#net = embed_net(args.low_dim, n_class, drop = args.drop, arch=args.arch)
# net = embed_net(args.low_dim, n_class, drop = args.drop, arch=args.arch,with_se=args.with_se,with_rpp=args.with_rpp,use_cbam=args.use_cbam,reduction=args.reduction)
net = embed_net(args.low_dim, n_class, drop=args.drop, arch=args.arch, neck=args.with_model_neck,
                    with_se=args.with_se, with_rpp=args.with_rpp, use_cbam=args.use_cbam, reduction=args.reduction)
net.to(device)    
cudnn.benchmark = True
net_dict=net.state_dict()
print('==> Resuming from checkpoint..')
checkpoint_path = args.model_path
#print(net.state_dict().keys())

if len(args.resume)>0:   
    model_path = checkpoint_path + args.resume
    print(model_path)
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        #print(checkpoint['net'].keys())
        pretrained_dict={k:v for k,v in checkpoint['net'].items() if k in net_dict}
        print(pretrained_dict.keys())
        net_dict.update(pretrained_dict)
        #print(pretrained_dict.keys())
        start_epoch = checkpoint['epoch']
        net.load_state_dict(net_dict)

        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))


if args.method =='id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

if dataset =='sysu':
    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode = args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, trial = 0)

      
elif dataset =='regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    
    # testing set
    query_img, query_label = process_test_regdb(data_path, trial = args.trial, modal = 'visible')
    gall_img, gall_label  = process_test_regdb(data_path, trial = args.trial, modal = 'thermal')
    
    gallset  = TestData(gall_img, gall_label, transform = transform_test, img_size =(args.img_w,args.img_h))
    gall_loader  = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
nquery = len(query_label)
ngall = len(gall_label)
print("Dataset statistics:")
print("  ------------------------------")
print("  subset   | # ids | # images")
print("  ------------------------------")
print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
print("  ------------------------------")

queryset = TestData(query_img, query_label, transform = transform_test, img_size =(args.img_w, args.img_h))   
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
print('Data Loading Time:\t {:.3f}'.format(time.time()-end))

feature_dim = args.low_dim

if args.arch =='resnet50':
    pool_dim = 2048
elif args.arch =='resnet18':
    pool_dim = 512
elif args.arch=='pcb_rpp':
    pool_dim=2048
elif args.arch=='cbam':
    pool_dim=2048


def extract_gall_feat(gall_loader):
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, feature_dim))
    gall_feat_pool = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            pool_feat, feat = net(input, input, test_mode[0])
            gall_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            gall_feat_pool[ptr:ptr+batch_num,: ] = pool_feat.detach().cpu().numpy()
            ptr = ptr + batch_num
            # label_g.append(label)
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat, gall_feat_pool

def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, feature_dim))
    query_feat_pool = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            pool_feat, feat = net(input, input, test_mode[1])
            query_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            query_feat_pool[ptr:ptr+batch_num,: ] = pool_feat.detach().cpu().numpy()
            ptr = ptr + batch_num
            # label_q.append(label)
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat, query_feat_pool 
    
query_feat, query_feat_pool = extract_query_feat(query_loader)    

all_cmc = 0
all_mAP = 0 
all_cmc_pool = 0
output=[]
labels=[]
if dataset =='regdb':
    gall_feat, gall_feat_pool = extract_gall_feat(gall_loader)
    # fc feature 
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    cmc, mAP  = eval_regdb(-distmat, query_label, gall_label)
    
    # pool5 feature
    distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
    cmc_pool, mAP_pool = eval_regdb(-distmat_pool, query_label, gall_label)

    print ('Test Trial: {}'.format(args.trial))
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19]))
    print('mAP: {:.2%}'.format(mAP))
    print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
    print('mAP: {:.2%}'.format(mAP_pool))
    
elif dataset =='sysu':
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, trial = trial)
        
        trial_gallset = TestData(gall_img, gall_label, transform = transform_test,img_size =(args.img_w,args.img_h))
        trial_gall_loader  = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        
        gall_feat, gall_feat_pool = extract_gall_feat(trial_gall_loader)
        
        # fc feature 
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        cmc, mAP  = eval_sysu(-distmat, query_label, gall_label,query_cam, gall_cam)
        
        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool = eval_sysu(-distmat_pool, query_label, gall_label,query_cam, gall_cam)
        if trial ==0:
            all_cmc = cmc
            all_mAP = mAP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
        
        print ('Test Trial: {}'.format(trial))
        print ('Test Trial: {}'.format(trial),file=test_log_file)

        print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19]))
        print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19]),file=test_log_file)

        print('mAP: {:.2%}'.format(mAP))
        print('mAP: {:.2%}'.format(mAP),file=test_log_file)

        print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
        print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]),file=test_log_file)

        print('mAP: {:.2%}'.format(mAP_pool))
        print('mAP: {:.2%}'.format(mAP_pool),file=test_log_file)

        uni_que=np.unique(query_label)
        uni_gall = np.unique(gall_label)
        # query_label=query_label
        # gall_label=gall_label
        n_q,n_g=0,0
        for i in range(20):
            label_id = uni_que[randrange(0, len(uni_que))]
            index_q=[i for i,a in enumerate(query_label) if a==label_id]
            index_g = [i for i,a in enumerate(gall_label) if a==label_id]
            output.extend(np.concatenate((query_feat[index_q, :], gall_feat[index_g, :]), 0))
            tmp_q=query_label[index_q]
            tmp_g=gall_label[index_g]
            n_q+=len(tmp_q)
            n_g += len(tmp_g)
            labels.extend(np.concatenate((tmp_q,tmp_g),0))
        # output = np.concatenate((query_feat[:100,:], gall_feat[:100,:]), 0)
        # labels = query_label.tolist()[:100]+gall_label.tolist()[:100]
        figure = draw_cluster(output, labels, n_q, n_g)
        #
        suffix_id = 'gloid_expcenter_1m0.05_pcb_cbam'
        plt.savefig(
            '/home/zhangc/projects/cross_modality_V/BDTR_modal_loss_cbam_debug/image/cluster/' +str(trial)+ '_test_' + suffix_id + '.jpg')
        plt.show()
        plt.pause(1)
        plt.close()

    cmc = all_cmc /10 
    mAP = all_mAP /10

    cmc_pool = all_cmc_pool /10 
    mAP_pool = all_mAP_pool /10
    print ('All Average:')
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]))
    print('mAP: {:.2%}'.format(mAP))
    print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
    print('mAP: {:.2%}'.format(mAP_pool))

