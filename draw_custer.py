from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import torch

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted') #调色板颜色温和
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})




def label2indis(label):
    indis = {}
    ind_list=[]
    count = -1
    for i, value in enumerate(label):

        if value not in indis:
            count += 1
            indis[value] = count

        ind_list.append(count)

    return indis,ind_list




#
# def draw_cluster(data,label,len_q,len_g):
#     tsne=TSNE(n_components=2,init='pca',random_state=0)
#
#     if torch.is_tensor(data):
#         result=tsne.fit_transform(data.cpu().data.numpy())
#     else:
#         result=tsne.fit_transform(data)
#     if torch.is_tensor(label):
#         label=label.cpu().data.numpy()
#     # cmaplist = [c for c in colors.cnames]
#     palette = np.array(sns.color_palette("hls", len_q+len_g))
#
#     indis=label2indis(label)
#     fig=plt.figure()
#     print("query:",len_q)
#     print("gallery:",len_g)
#     for i in range(len_q+len_g):
#         # color=plt.cm.Set1(indis[label[i]])
#         color=cmaplist[indis[label[i]]+3]
#
#         if i<len_q:
#             mol1=plt.scatter(result[i,0],result[i,1],c=color,marker='.')
#         else:
#             mol2=plt.scatter(result[i,0],result[i,1],c=color,marker='*')
#     plt.legend([mol1,mol2],['modality 1','modality 2'],loc='upper right')
#
#     return fig


def draw_cluster(data, label,len_q,len_g,num_label):
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    if torch.is_tensor(data):
        result = tsne.fit_transform(data.cpu().data.numpy())
    else:
        result = tsne.fit_transform(data)
    if torch.is_tensor(label):
        label = label.cpu().data.numpy()
    # cmaplist = [c for c in colors.cnames]

    palette = np.array(sns.color_palette("hls", num_label+5))
    _,indis=label2indis(label)
    _,indis_q = label2indis(label[0:len_q])
    _,indis_g = label2indis(label[len_q:len_q + len_g])
    fig = plt.figure()
    print("query:", len_q)
    print("gallery:", len_g)
    mol1 = plt.scatter(result[0:len_q, 0], result[0:len_q, 1], c=palette[indis_q], marker='.')

    mol2 = plt.scatter(result[len_q:len_q + len_g, 0], result[len_q:len_q + len_g, 1], c=palette[indis_g], marker='*')
    plt.legend([mol1, mol2], ['modality 1', 'modality 2'], loc='upper right',frameon=True,facecolor='white')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    return fig
