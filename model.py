import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from senet.se_resnet import se_resnet50, SEBottleneck
from senet import resnet
from senet.SCPNet import get_scp_model
import numpy as np
from senet.cbam import *
from senet.cbam_fu import *
import math
from torch.nn import Parameter
import torch.nn.functional as F
from senet.grad_cbam import *


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out
'''
class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
'''
# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        #print(m.bias)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
        # nn.init.zeros_(m.bias.data)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|

class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)] 
        feat_block += [nn.BatchNorm1d(low_dim)]
        
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block
    def forward(self, x):
        x = self.feat_block(x)
        return x
        
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []       
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier
    def forward(self, x):
        x = self.classifier(x)
        return x

class ClassBlock_ArcMargin(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ClassBlock_ArcMargin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class ClassBlock_pcb(nn.Module):
    def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
        super(ClassBlock_pcb, self).__init__()
        add_block = []

        add_block += [nn.Conv2d(input_dim, num_bottleneck, kernel_size=1, bias=False)]
        add_block += [nn.BatchNorm2d(num_bottleneck)]
        if relu:
            #add_block += [nn.LeakyReLU(0.1)]
            add_block += [nn.ReLU(inplace=True)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x_feat = torch.squeeze(x)
        x = self.classifier(x_feat)
        return x,x_feat

# Define the ResNet18-based Model
class visible_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18',with_se=False,with_rpp=False,use_cbam=False,class_num=395,reduction=16):
        super(visible_net_resnet, self).__init__()
        if arch =='resnet18':
            print("visible_net with resenet18 architecture setting.....")
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet50':
            print("visible_net with resenet50 architecture setting.....")
            # model_ft = resnet.resnet50(pretrained=True,with_se=with_se)
            model_ft = resnet.pcb_rpp(pretrained=True, with_rpp=with_rpp, use_cbam=use_cbam, class_num=class_num)
        #add by zc
        elif arch=='se_resnet50':
            print("visible_net with se_resnet50 architecture setting.....")
            model_ft=se_resnet50(pretrained=True)

        elif arch=='pcb_rpp':
            model_ft=resnet.pcb_rpp(pretrained=True,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num)

        elif arch=='pcb_pyramid':
            model_ft=resnet.pcb_rpp(pretrained=True,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num)

        elif arch=='cbam':
            model_ft=resnet.resnet50(pretrained=True,use_cbam=use_cbam)

        elif arch=='scpnet':
            model_ft=get_scp_model(pretrained=True,nr_class=class_num)
        ##end by zc
        # avg pooling to global pooling
        # if arch == 'pcb_rpp':
        #     if with_rpp:
        #         model_ft.avgpool = resnet.RPP()
        #         print("-------RPP module in visible starting------")
        #     else:
        #         model_ft.avgpool = nn.AdaptiveAvgPool2d((6, 1))
        #         print("-------No RPP module in visible------")
        # else:
        #     model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.visible = model_ft
        self.backbone=model_ft.backbone
        self.dropout = nn.Dropout(p=0.5)
        #add by zc
        self.with_se=with_se
        self.arch=arch
        self.feature = FeatureBlock(2048, 512, dropout=0.5)
        '''
        if self.with_se:
            self.layer4=self.visible._make_layer(SEBottleneck, 512, 3, stride=2)
        else:
            self.layer4=self.visible.layer4()
        '''
        ##end by zc
    def forward(self, x):

        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        x = self.visible.layer3(x)
        x = self.visible.layer4(x)

        #x = self.visible.avgpool(x)
        #x = x.view(x.size(0), x.size(1))
            # x = self.dropout(x)
        return x

class thermal_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18',with_se=False,with_rpp=False,use_cbam=False,class_num=395,reduction=16):
        super(thermal_net_resnet, self).__init__()
        if arch =='resnet18':
            print("thermal_net with resenet18 architecture setting.....")
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet50':
            print("thermal_net with resenet50 architecture setting.....")
            # model_ft = resnet.resnet50(pretrained=True,with_se=with_se)
            model_ft = resnet.pcb_rpp(pretrained=True, with_rpp=with_rpp, use_cbam=use_cbam, class_num=class_num)
        #add by zc
        elif arch=='se_resnet50':
            print("thermal_net with se_resnet50 architecture setting.....")
            model_ft=se_resnet50(pretrained=True)

        elif arch=='pcb_rpp':
            model_ft=resnet.pcb_rpp(pretrained=True,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num)

        elif arch=='pcb_pyramid':
            model_ft=resnet.pcb_rpp(pretrained=True,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num)

        elif arch=='cbam':
            model_ft=resnet.resnet50(pretrained=True,use_cbam=use_cbam)

        elif arch=='scpnet':
            model_ft=get_scp_model(pretrained=True,nr_class=class_num)
        # avg pooling to global pooling

        self.thermal = model_ft
        self.backbone = model_ft.backbone
        self.dropout = nn.Dropout(p=0.5)
        #add by zc
        self.with_se=with_se
        self.arch=arch
        self.feature = FeatureBlock(2048, 512, dropout=0.5)
            ##end by zc

    def forward(self, x):

        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        x = self.thermal.layer2(x)
        x = self.thermal.layer3(x)
        x = self.thermal.layer4(x)


        #x = self.thermal.avgpool(x)

        #x = x.view(x.size(0), x.size(1))
        return x
        # x = self.dropout(x)


class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop = 0.5, arch ='resnet50',neck='no',with_se=False,with_rpp=False,use_cbam=False,reduction=16):#param: with_se:whether use SEModule ,reduction is the ratio  add by zc
        super(embed_net, self).__init__()
        self.num_part=6

        self.out_pool_size = [1, 3, 6]
        self.total_pool = np.sum(self.out_pool_size)
        if arch =='resnet18':
            self.visible_net = visible_net_resnet(arch = arch,with_se=with_se,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num,reduction=reduction)
            self.thermal_net = thermal_net_resnet(arch = arch,with_se=with_se,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num,reduction=reduction)
            pool_dim = 512
        elif arch =='resnet50':

            self.visible_net = visible_net_resnet(arch = arch,with_se=with_se,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num,reduction=reduction)
            self.thermal_net = thermal_net_resnet(arch = arch,with_se=with_se,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num,reduction=reduction)
            pool_dim = 2048
        #write by zc
        elif arch=='se_resnet50':
            self.visible_net = visible_net_resnet(arch = arch,with_se=with_se,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num,reduction=reduction)
            self.thermal_net = thermal_net_resnet(arch = arch,with_se=with_se,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num,reduction=reduction)
            pool_dim = 2048
        elif arch=='pcb_rpp':
            print("using PCB_RPP net")
            self.visible_net = visible_net_resnet(arch = arch,with_se=with_se,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num,reduction=reduction)
            self.thermal_net = thermal_net_resnet(arch = arch,with_se=with_se,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num,reduction=reduction)
            pool_dim = class_num*self.num_part  #206*6  #512*6
        elif arch=='pcb_pyramid':
            print("using PCB_pyramid net")
            self.visible_net = visible_net_resnet(arch=arch, with_se=with_se, with_rpp=with_rpp, use_cbam=use_cbam,
                                                  class_num=class_num, reduction=reduction)
            self.thermal_net = thermal_net_resnet(arch=arch, with_se=with_se, with_rpp=with_rpp, use_cbam=use_cbam,
                                                  class_num=class_num, reduction=reduction)
            pool_dim = class_num * np.sum(self.out_pool_size)  # 206*6  #512*6
        elif arch=='cbam':
            print("using CBAM net")
            self.visible_net = visible_net_resnet(arch = arch,with_se=with_se,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num,reduction=reduction)
            self.thermal_net = thermal_net_resnet(arch = arch,with_se=with_se,with_rpp=with_rpp,use_cbam=use_cbam,class_num=class_num,reduction=reduction)
            pool_dim = 2048

        self.feature = FeatureBlock(pool_dim, low_dim, dropout = drop)
        self.glo_feature=FeatureBlock(2048, low_dim, dropout = drop)
        self.l2norm = Normalize(2)
        self.arch=arch
        self.neck=neck
        if self.neck == 'no':

            self.classifier = ClassBlock(low_dim, class_num, dropout=drop)
            # print("using Arcmargin loss")
            # self.classifier=ClassBlock_ArcMargin(low_dim, class_num, s=30, m=0.5, easy_margin=True)
            # print("using add_margin loss")
            # self.classifier=AddMarginProduct(low_dim, class_num, s=30, m=0.35)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(low_dim)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(low_dim, class_num, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

        if use_cbam:
            print("using cbam")
            self.attention=CBAM(2048, 16,pool_types=['max'])#'avg', 'max'
            # self.pyramid_attention=pyramid_attention(self.attention)
        else:
            self.attention = None
            self.pyramid_attention=None

        self.glo_classifier = ClassBlock_pcb(2048, class_num, True, 512)
        if arch == 'pcb_rpp':
            # define 6 classifiers
            self.pcb_classifier = nn.ModuleList()
            # self.attention=nn.ModuleList()
            for i in range(self.num_part):
                self.pcb_classifier.append(ClassBlock_pcb(2048, class_num, True, 512))

            self.classifier = self.pcb_classifier

            self.pcb_avgpool = nn.AdaptiveAvgPool2d((self.num_part, 1))
            self.pcb_maxpool=nn.AdaptiveMaxPool2d((self.num_part,1))
            self.avgpool =self.pcb_avgpool



        elif arch=='pcb_pyramid':
            self.spp=self.make_spp(out_pool_size=self.out_pool_size)
            self.pcb_pyr_classifier = nn.ModuleList()
            # self.attention = nn.ModuleList()
            for i in range(np.sum(self.out_pool_size)):
                self.pcb_pyr_classifier.append(ClassBlock_pcb(2048, class_num, True, 512))

            self.classifier = self.pcb_pyr_classifier
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.glo_avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, modal =0):
        x1=self.visible_net(x1)
        x2=self.thermal_net(x2)
        if self.attention is not None:

            # out1 = self.attention(x1)
            # out2 = self.attention(x2)
            #
            # x1 = self.l2norm(out1 + x1)
            # x2 = self.l2norm((out2 + x2))
            #
            # print('pyramid attention')
            x1 = self.pyramid_attention(x1)
            x2 = self.pyramid_attention(x2)


        v_global_feat=self.glo_avgpool(x1)#.view(x1.size(0),-1)
        t_global_feat = self.glo_avgpool(x2)#.view(x2.size(0), -1)

        if self.arch=='pcb_rpp':
            # x1 = self.pcb_avgpool(x1)
            # x2 = self.pcb_avgpool(x2)
            #
            # x1=nn.AdaptiveMaxPool2d()
            # v_global_feat=x1.view(x1.size(0),-1)
            # t_global_feat = x2.view(x2.size(0), -1)

            f_v,feat_v,feat_vpg,feat_vp= self.extract_pcb_feature(x1)
            f_t,feat_t,feat_tpg,feat_tp= self.extract_pcb_feature(x2)

            out={}
            feat_p={}
            if modal==0:
            #f_v,feat_v = self.visible_net(x1)
            # f_t,feat_t = self.thermal_net(x2)
                for i in range(6):
                    out[i] = torch.cat((f_v[i], f_t[i]), 0)
                    feat_p[i]=torch.cat((feat_vp[i], feat_tp[i]), 0)
                x = torch.cat((feat_v, feat_t), 0)
                # global_feat = self.feature(x)
                # x = torch.cat((feat_vpg, feat_vpg), 0)
            # get six part feature batchsize*2048*6
            elif modal ==1:
                out=f_v
                x=feat_v
                v_g=v_global_feat
                t_g = t_global_feat
                # x=feat_vpg
                # global_feat = self.glo_feature(x)
            elif modal ==2:
                out=f_t
                x=feat_t
                v_g = v_global_feat
                t_g = t_global_feat
                # x=feat_tpg
                # global_feat = self.glo_feature(x)

            global_feat = self.feature(x)
            v_global_out,v_global_feat=self.glo_classifier(v_global_feat)
            t_global_out,t_global_feat = self.glo_classifier(t_global_feat)
            # v_global_feat = feat_vpg
            # t_global_feat = feat_tpg

            if self.training:
                return [out,v_global_out,t_global_out,feat_p], self.l2norm(global_feat)
            else:
                return self.l2norm(x), self.l2norm(global_feat)

        elif self.arch=='pcb_pyramid':

            f_v,feat_v,feat_vpg,feat_vp=self.extract_pyramid_pcb(x1)
            f_t,feat_t,feat_tpg,feat_tp=self.extract_pyramid_pcb(x2)
            out={}
            feat_p = {}
            if modal == 0:
                for i in range(np.sum(self.out_pool_size)):
                    out[i]=torch.cat((f_v[i], f_t[i]),0)
                    feat_p[i] = torch.cat((feat_vp[i], feat_tp[i]), 0)
                x = torch.cat((feat_v, feat_t), 0)
                # get six part feature batchsize*2048*6
            elif modal == 1:
                out = f_v
                x = feat_v
            elif modal == 2:
                out = f_t
                x = feat_t
            global_feat = self.feature(x)
            v_global_out, v_global_feat = self.glo_classifier(v_global_feat)
            t_global_out, t_global_feat = self.glo_classifier(t_global_feat)
            if self.training:
                return [out,v_global_out,t_global_out,feat_p], self.l2norm(global_feat)
            else:
                return self.l2norm(x), self.l2norm(global_feat)
        else:
            xv = self.glo_avgpool(x1)
            xt = self.glo_avgpool(x2)

            if modal==0:
                xv =xv.view(xv.size(0),xv.size(1))
                xt = xt.view(xt.size(0),xt.size(1))
                x = torch.cat((xv,xt), 0)
            elif modal ==1:
                x = xv.view(xv.size(0),xv.size(1))
            elif modal ==2:
                x = xt.view(xt.size(0),xt.size(1))

            global_feat = self.feature(x)

            if self.neck == 'no':
                feat = global_feat
            elif self.neck == 'bnneck':
                feat = self.bottleneck(global_feat)  # normalize for angular softmax
            out = self.classifier(feat)

            if self.training:
                return out,self.l2norm(feat)#out, self.l2norm(global_feat)
            else:
                return self.l2norm(x2), self.l2norm(feat)



    def extract_pcb_feature(self,x):

        part = {}
        predict = {}
        predict_feat={}
        step=math.ceil(x.size(2)/self.num_part)
        # get six part feature batchsize*2048*6
        for i in range(self.num_part):
            part[i] = x[:, :, i*step:(i+1)*step,:]
            predict[i],predict_feat[i] = self.classifier[i](self.glo_avgpool(part[i]))

        y = []
        feat = []
        x1_ff = torch.FloatTensor().cuda()#torch.zeros_like(predict[0]).cuda()
        x2_ff = torch.FloatTensor().cuda()

        for i in range(6):
            y.append(predict[i])
            x1_ff=torch.cat((x1_ff,predict[i]),1)
            x2_ff = torch.cat((x2_ff, self.l2norm(predict_feat[i])), 1)
            feat.append(self.l2norm(predict_feat[i]))

        return y,x1_ff,x2_ff,feat #y,x2_ff

    def make_spp(self,out_pool_size=[1,4,8]):
        func = []

        for i in range(len(out_pool_size)):
            func.append(nn.AdaptiveAvgPool2d(output_size=(out_pool_size[i], 1)))

        return func


    def extract_pyramid_pcb(self,x):
        spp = []
        feat = []
        part={}
        predict = {}
        predict_feat = {}
        x1_ff = torch.FloatTensor().cuda()  # torch.zeros_like(predict[0]).cuda()
        x2_ff = torch.FloatTensor().cuda()
        tmp=0
        for i in range(len(self.out_pool_size)):
            for j in range(self.out_pool_size[i]):
                part[tmp+j]=self.spp[i](x)[:, :, j, :]
                predict[tmp+j], predict_feat[tmp+j] =self.classifier[tmp+j](torch.unsqueeze(part[tmp+j],3))
                spp.append(predict[tmp+j])
                x1_ff = torch.cat((x1_ff,predict[tmp+j]), 1)
                x2_ff = torch.cat((x2_ff, predict_feat[i]), 1)
                feat.append(self.l2norm(predict_feat[tmp+j]))
            tmp +=self.out_pool_size[i]
        glo_feat=torch.cat(spp,1)


        return spp,x1_ff,x2_ff,feat

    def pyramid_attention(self,x):
        part_3 = {}
        part_6 = {}

        step_3 = math.ceil(x.size(2) / 3)
        step_6 = math.ceil(x.size(2) / 6)
        # = out + part_3[i]

        for i in range(6):
            part_6[i] = x[:, :, i * step_6:(i + 1) * step_6, :]
            if self.attention is not None:
                part_6[i] = self.attention(part_6[i])
                # part_6[i] = part_6_x[i]* part_6[i]+ part_6_x[i]
        for i in range(3):
            tmp_x = x[:, :, i * step_3:(i + 1) * step_3, :]
            part_3[i] = tmp_x * torch.cat((part_6[2 * i], part_6[2 * i + 1]), 2) + tmp_x
            if self.attention is not None:
                part_3[i] = self.attention(part_3[i])
        glo_att_feature = torch.cat((part_3[0], part_3[1]), 2)
        glo_att_feature = x* torch.cat((glo_att_feature, part_3[2]), 2) + x
        glo_att_feature = self.attention(glo_att_feature) + glo_att_feature

        return self.l2norm(glo_att_feature)

# class pyramid_attention(nn.Module):
#     def __init__(self,attention):
#         super(pyramid_attention, self).__init__()
#         self.attention=attention
#
#     def forward(self,x):
#         part_3 = {}
#         part_6 = {}
#
#         step_3 = math.ceil(x.size(2) / 3)
#         step_6 = math.ceil(x.size(2) / 6)
#         # = out + part_3[i]
#
#         for i in range(6):
#             part_6[i] = x[:, :, i * step_6:(i + 1) * step_6, :]
#             if self.attention is not None:
#                 part_6[i] = self.attention(part_6[i])
#                 # part_6[i] = part_6_x[i]* part_6[i]+ part_6_x[i]
#         for i in range(3):
#             tmp_x = x[:, :, i * step_3:(i + 1) * step_3, :]
#             part_3[i] = tmp_x * torch.cat((part_6[2 * i], part_6[2 * i + 1]), 2) + tmp_x
#             if self.attention is not None:
#                 part_3[i] = self.attention(part_3[i])
#         glo_att_feature = torch.cat((part_3[0], part_3[1]), 2)
#         glo_att_feature = x* torch.cat((glo_att_feature, part_3[2]), 2) + x
#         glo_att_feature = self.attention(glo_att_feature) + glo_att_feature
#         l2norm = Normalize(2)
#         return l2norm(glo_att_feature)