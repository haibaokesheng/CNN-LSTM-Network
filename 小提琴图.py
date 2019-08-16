# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:41:16 2019

@author: liusiyan
"""
import numpy as np
import scipy.io as sio  
import matplotlib.pyplot as plt
from my_functions import slide_window
#import seaborn as sns
# 重要模型参数
signal_len = 4     # 信号长度 (s)
Fs = 250
# 加载数据
data = sio.loadmat(r'.\ECG_data\public_ECG_5fold_data_with_annotion.mat')['data']
ecg = np.vstack(data[:,0])
Y = np.vstack(data[:,1]).reshape(-1)
original_label = np.vstack(data[:,3]).reshape(-1)
# 取前 signal_len 的信号
ecg = ecg[:,0:int(signal_len*Fs)]

# 10:正常,19:VF,20:VFL,21:VT
NshR_index = np.argwhere(Y==0).reshape(-1)
#NshR_index = np.argwhere(original_label==24).reshape(-1)

VF_index = np.argwhere(original_label==20).reshape(-1) # 室颤
VT_index = np.argwhere(original_label==21).reshape(-1) # 室速

NshR_num = 520
#NshR_num = 0

NshR_label = original_label[NshR_num]
VF_num = 25
VT_num = 10


# 静态斜率
NshR_cnn  = np.diff(ecg[NshR_index[NshR_num],:])
VF_cnn = np.diff(ecg[VF_index[VF_num],:])
VT_cnn = np.diff(ecg[VT_index[VT_num],:])
cnn_ecg = np.vstack([NshR_cnn,VF_cnn,VT_cnn])

# 动态斜率
ecg_w = slide_window(ecg,window=0.6,step=0.2)[0]
ecg_w = np.diff(ecg_w)  

NshR_w = ecg_w[NshR_index[NshR_num],:,:]
VF_w = ecg_w[VF_index[VF_num],:,:]
VT_w = ecg_w[VT_index[VT_num],:,:]

dynamic = np.dstack([NshR_w,VF_w,VT_w])
dynamic = np.transpose(dynamic,[1,0,2])

t = np.arange(1,Fs*signal_len)/Fs

min_y = -0.105
max_y = 0.095

plt.figure(figsize=(22,8.5))
from matplotlib.gridspec import GridSpec

gs = GridSpec(3,15)
name = ['NSR (mV)','VF (mV)','VT (mV)']
for i in range(3):
    ax = plt.subplot(gs[i,0:2])
    plt.plot(t,cnn_ecg[i,:],color='black',linewidth = 0.5) 
    plt.xlim([0,4])
    plt.ylim([min_y,max_y])
    plt.xticks([0,1,2,3,4])
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    #if(i==0):
    #    plt.title('ECG filterd by Conv layer',fontsize=16)
    if(i==2):
        plt.xlabel('Time (s)\n\n (a)',fontsize=16)
    plt.ylabel(name[i],fontsize=16)
    
for i in range(3):                 # scale = 'area',测度小提琴图的宽度：area-面积相同，count-按照样本数量决定宽度，width-宽度一样
    ax = plt.subplot(gs[i,3:13])   # gridsize = 40, 设置小提琴图边线的平滑度，越高越平滑
                                   # inner = 'box', 设置内部显示类型 → “box”, “quartile”, “point”, “stick”, None
    #sns.violinplot(data=dynamic[:,:,i],linewidth = 1.8,width = 0.5,palette ='hls',scale='area',gridsize = 40,inner = 'quartile',)
    #sns.violinplot(data=dynamic[:,:,i],linewidth = 1.8,width = 0.5,color='white',scale='area',gridsize = 40,inner = 'box',)
    ax.violinplot(dataset=dynamic[:,:,i],showmeans=False,showmedians=False)
    plt.xlim([0.5,18.5])
    plt.ylim([min_y,max_y])
    plt.xticks([i for i in range(1,19)])
    plt.xticks(fontsize=16)
    plt.yticks([])
    #if(i==0):
    #    plt.title('The amplitude distrbution of the sliding windows',fontsize=16)
    if(i==2):
        plt.xlabel('Index of the sliding windows\n\n(b)',fontsize=16)
for i in range(3):
    ax = plt.subplot(gs[i,14])                                           
   # sns.violinplot(data=cnn_ecg[i,:],linewidth = 1.8,width = 0.5,scale='area',gridsize = 40,inner = 'quartile',color='white')
    ax.violinplot(dataset=cnn_ecg[i,:],showmeans=False,showmedians=False)
    plt.ylim([min_y,max_y])
    plt.xticks([])
    plt.yticks([])
    if(i==2):
        plt.xlabel('\n\n\n(c)',fontsize=16)
plt.savefig(r'.\小提琴图_论文.jpeg') 
plt.show()

print(len(NshR_index))