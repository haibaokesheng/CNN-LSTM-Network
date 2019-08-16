# -*- coding: utf-8 -*-
"""
可除颤心律信号识别
卷积+顺序池化+LSTM网络
2019年8月15日
错误样本分析
"""
import pickle
import numpy as np
import scipy.io as sio  
import matplotlib.pyplot as plt

#from keras.layers import Dense,Input,Lambda,Conv2D,LSTM,concatenate
#from keras.models import Model
#from keras import backend as K
#import tensorflow as tf
from my_functions import slide_window,calc_ber
from my_model import my_model_8s,my_model_4s

Ber = []
Error = []
# 重要模型参数
signal_len = 2     # 信号长度 (s)
window  = 0.6      # 滑动窗宽(s)
step = 0.2         # 滑动窗移动步长      
Fs = 250           # 信号采样率

# 加载数据
data = sio.loadmat(r'.\ECG_data\public_ECG_5fold_data_with_annotion_1.mat')['data']

annotation = [21,19,20,2,5,6,8,10,11,13,15,16,20] # 论文中的顺序
name = ['VT','VF','VFL','AFIB','B','BI','HGEA','NSR','NOD','PM','SBR','SVTA','VER']   

original_label = np.vstack(data[:,3]).reshape(-1)

cv = np.eye(5) #1是测试集  0是训练集
for k in range(len(cv)):   
    # 测试集索引
    index = np.argwhere(cv[k,]==1).reshape(-1)
    ecg = np.vstack(data[index,0])
    Y_test = np.vstack(data[index,1]).reshape(-1)
    # 取前 signal_len 的信号
    ecg = ecg[:,0:int(signal_len*Fs)]
    # 获取滑动窗数据
    X_test = slide_window(ecg,window,step)[0]
    # 增加一维度以适应二维卷积的输入格式
    X_test = np.expand_dims(X_test,axis=-1)
    # 加载模型拓扑结构
    if(signal_len == 6 or signal_len == 8):
        model = my_model_8s(X_test)
        path = 'CNN_SP_LSTM\\CNN_SP_LSTM_units=50'  
    else:
        model = my_model_4s(X_test)
        path = 'CNN_SP_LSTM\\double_stream\\units =20'
    # 加载训练好的模型权重
    model.load_weights('.\model\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.h5py')
    # 预测
    prob_testY = model.predict(X_test)  
    # 加载重要参数
    data_path = r'.\important_data\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
    with open(data_path,'rb') as f:
        parameter = pickle.load(f)
        thr = parameter['threshold']
    # 概率转换成0-1标签
    pred_testY = (prob_testY >= thr).astype('int8') 
    pred_testY = pred_testY.reshape(-1)
    # 找出错误样本的索引
    Ber.append(calc_ber(Y_test,pred_testY)[2])
    error_index = np.argwhere((pred_testY-Y_test)!=0).reshape(-1)
    Error.append(error_index)

error = np.hstack([Error[0],Error[1],Error[2],Error[3],Error[4]])
Acc = np.zeros([13])
for i,element in enumerate (annotation):
    #print(i)
    one = np.argwhere(original_label==element).reshape(-1) # 423 AF index
    # 错误样本索引与当前心律的索引的交集
    c = list(set(error).intersection(set(one)))
    Acc[i] = (len(one)-len(c))/len(one)
    #Acc.append(acc)
    print(name[i],':%.3f'%Acc[i])
    
import pandas as pd
df = pd.DataFrame(Acc,index =name)  
df = df.transpose()
print(df)
print('Ber:%.2f%%'%np.mean(Ber)) 


# 画图
#    for num in error_index:
#        plt.figure(figsize=(8,5))
#        plt.plot(ecg[num,:],linewidth = 2)
#        plt.title('prob:%.2f,thr:%.2f,pred_label:%d,original_label:%d'
#                  %(prob_testY[num],thr,pred_testY[num],original_label[num]),fontsize=18)
#        print(num)
#        plt.show()
    # NshR
#    for i in annotation:
#        ecg_index = np.argwhere(original_label==i).reshape(-1) 
#        ecg = X_test[ecg_index,:,:,:]
#    
    
    
        
  
