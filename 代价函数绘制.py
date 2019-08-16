# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 20:17:56 2019

@author: liusiyan
"""
import pickle
import matplotlib.pyplot as plt
import math
train_loss = []
path = []
path.append('CNN_SP_LSTM\\double_stream\\units =20') # 2s最佳结果的路径
path.append('CNN_SP_LSTM\\double_stream\\units =20') 
path.append('CNN_SP_LSTM\\CNN_SP_LSTM_units=20') 
path.append('CNN_SP_LSTM\\CNN_SP_LSTM_units=20') 

signal_len = [2,4,6,8]
for i in range(4):
    for k in range(5):
        data_path = '.\important_data\\'+path[i]+'\\'+str(signal_len[i])+'s-train_'+str(k+1)+'-fold.pickle'
        with open(data_path, 'rb') as f:
            a = pickle.load(f)
            train_loss.append(a['train_loss'])
            
font = {'family':'Times New Roman','weight':'normal', 'size':19}

marker = ['','x','o','^','v']
color = ['red','b','black','cyan','m']
x = list(range(1,101))
c = 10
# 代价函数对数曲线坐标轴
fig = plt.figure(figsize=(c,0.65*c))
for i in range(4):
    plt.subplot(2,2,i+1)
    for j in range(i*5,(i+1)*5):
        y = [math.log(k,2) if k>=2 else k for k in train_loss[j]]
        plt.plot(x,y,linewidth=0.8,label='fold %d'%(j%5+1),color=color[j%5],marker=marker[j%5],markersize=4)
    plt.xlim([-1,100])   
    plt.legend(prop=font,loc='upper right')
    plt.xticks(fontsize = 16)
    plt.yticks([1e-8,1,2,3,4,5,6],['$0$','$1$','$2$','$4$','$8$','$16$','$32$','$64$'],fontsize =16)
    if(i==3):
        plt.yticks([1e-8,1,2,3,4,5],['$0$','$1$','$2$','$4$','$8$','$16$','$32$'],fontsize =16)
    if(i==2 or i==3):
        plt.xlabel('Epochs',font)
        plt.text(9,4.5,'%d-s ECG records'%signal_len[i],font)
    if(i==0 or i==2):
        plt.ylabel('Training loss',font)
    if(i==0 or i==1):
         plt.text(9,5,'%d-s ECG records'%signal_len[i],font)   
fig.tight_layout()
#plt.savefig('loss.png',dpi=300)
plt.show()
