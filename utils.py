import os
from collections import defaultdict
import numbers
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import scipy.io as scio

def GenIdx( train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)
        
    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos

    
class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, batchSize):        
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        
        sample_color = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        N = np.maximum(len(train_color_label), len(train_thermal_label))
        
        for j in range(int(N/batchSize)+1):
            batch_idx = np.random.choice(uni_label, batchSize, replace = False)
            
            for i in range(batchSize):
                sample_color[i]  = np.random.choice(color_pos[batch_idx[i]], 1)
                sample_thermal[i] = np.random.choice(thermal_pos[batch_idx[i]], 1)
            
            if j ==0:
                index1= sample_color
                index2= sample_thermal
            else:
                index1 = np.hstack((index1, sample_color))
                index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class RandomIdentitySampler_alignedreid(Sampler):
    """
        Randomly sample N identities, then for each identity,
        randomly sample K instances, therefore batch size is N*K.

        Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

          Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, batchSize, num_instances):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        self.num_instances=num_instances
        sample_color = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        self.N = np.maximum(len(train_color_label), len(train_thermal_label))
        self.num_identities=batchSize//self.num_instances
        for j in range(int(self.N/batchSize)+1):#int(len(uni_label)/self.num_identities)+1):
            batch_idx = np.random.choice(uni_label, self.num_identities, replace=False)

            for i in range(len(batch_idx)):

                t_sample_color = np.random.choice(color_pos[batch_idx[i]], self.num_instances, replace=False)
                t_sample_thermal= np.random.choice(thermal_pos[batch_idx[i]], self.num_instances, replace=False)
                for k in range(len(t_sample_color)):
                    sample_color[self.num_instances*i+k]=t_sample_color[k]
                # for s in range(len(t_sample_thermal)):
                    sample_thermal[self.num_instances*i+k]=t_sample_thermal[k]
            if j == 0:
                index1 = sample_color
                index2 = sample_thermal
            else:
                index1=np.hstack((index1, sample_color))
                index2 = np.hstack((index2, sample_thermal))
        self.index1 = index1
        self.index2 = index2
        # self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

class triplet_Sampler_reid(Sampler):
    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, batchSize,num_identities=8):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        sample_color = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        anchor_rgb_list=np.arange(batchSize)
        anchor_ir_list = np.arange(batchSize)
        self.N = np.maximum(len(train_color_label), len(train_thermal_label))
        self.num_identities = num_identities
        self.num_instances = batchSize // self.num_identities

        for j in range(int(self.num_instances*self.N / batchSize) + 1):
            batch_idx = np.random.choice(uni_label, self.num_identities, replace=False)

            for i in range(self.num_identities):
                anchor_rgb =np.random.choice(color_pos[batch_idx[i]], 1)
                anchor_ir = np.random.choice(thermal_pos[batch_idx[i]], 1)

                update_color_pos=[x for x in color_pos[batch_idx[i]] if x!=anchor_rgb]
                update_ir_pos = [x for x in thermal_pos[batch_idx[i]] if x != anchor_ir]
                tmp_sample_color = np.random.choice(update_color_pos, self.num_instances)
                tmp_sample_thermal = np.random.choice(update_ir_pos, self.num_instances)

                for k in range(len(tmp_sample_color)):
                    sample_color[self.num_instances*i+k]=tmp_sample_color[k]
                    anchor_rgb_list[self.num_instances*i+k]=anchor_rgb
                # for s in range(len(t_sample_thermal)):
                    sample_thermal[self.num_instances*i+k]=tmp_sample_thermal[k]
                    anchor_ir_list[self.num_instances * i + k] = anchor_ir
            if j == 0:
                index1 = sample_color
                index2 = sample_thermal
                anchor_pos_rgb=anchor_rgb_list
                anchor_pos_ir = anchor_ir_list
            else:
                index1 = np.hstack((index1, sample_color))
                index2 = np.hstack((index2, sample_thermal))
                anchor_pos_rgb =np.hstack(anchor_pos_rgb,anchor_rgb_list)
                anchor_pos_ir = np.hstack(anchor_pos_ir, anchor_ir_list)

        self.index1 = index1
        self.index2 = index2
        self.anchor_pos_rgb=anchor_pos_rgb
        self.anchor_pos_ir=anchor_pos_ir

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
