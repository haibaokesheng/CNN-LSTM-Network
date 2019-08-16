# -*- coding: utf-8 -*-
"""
可除颤心律信号识别
CNN3 作为特征提取器
2019.8.11
"""
import time;
time_start = time.time();

import warnings
warnings.filterwarnings("ignore")
import os
os.chdir('D:\project\python_code\ECG_project_2019')
import time;
time_start = time.time();
import pickle
import numpy as np
import scipy.io as sio  
from keras.layers import Dense,Input,Conv1D,MaxPooling1D,Flatten
from keras.models import Model
# 重要模型参数
signal_len = 2     # 信号长度 (s)
Fs = 250           # 信号采样率  
path ='CNNE3'
cv = np.eye(5) 
# 加载原始数据
data = sio.loadmat(r'.\ECG_data\public_ECG_5fold_data_with_annotion_1.mat')['data']
# 加载 1-12 Hz的数据,即VMED 的123
data_mved_123  = sio.loadmat(r'.\ECG_data\VMED_123\ECG_MVED_'+str(signal_len)+'s_123.mat')['data_ecg_MVED_123']
## 加载 13-17 Hz的数据  VMED 的67
data_mved_67  = sio.loadmat(r'.\ECG_data\VMED_67\ECG_MVED_'+str(signal_len)+'s_67.mat')['data_ecg_MVED_67']

for k in range(len(cv)):   #交叉验证
    # 训练集索引
    index = np.argwhere(cv[k,]==0).reshape(-1)
    X_train = np.vstack(data[index,0])             # 第 0 列是原始信号
    X_train_123 = np.vstack(data_mved_123[index,0])
    X_train_67 = np.vstack(data_mved_67[index,0])
    Y_train = np.vstack(data[index,1]).reshape(-1) # 第 1 列是标记信息
    # 测试集索引
    index = np.argwhere(cv[k,]==1).reshape(-1)
    X_test = np.vstack(data[index,0])
    X_test_123 = np.vstack(data_mved_123[index,0])
    X_test_67 = np.vstack(data_mved_67[index,0])
    Y_test = np.vstack(data[index,1]).reshape(-1)

    # 原始数据取前 signal_len 的信号
    X_train = X_train[:,0:int(signal_len*Fs)]
    X_test = X_test[:,0:int(signal_len*Fs)]
    X_train = np.expand_dims(X_train,axis=-1)
    X_test = np.expand_dims(X_test,axis=-1)

    # 信号拼接
    X_train = np.dstack([X_train,X_train_123,X_train_67])
    X_test = np.dstack([X_test,X_test_123,X_test_67])
    
    del X_train_123,X_test_123,X_train_67,X_test_67
    
    # 搭建网络
    input0 = Input(shape=(X_train.shape[1],X_train.shape[2]),name='input0')
    tensor = Conv1D(filters=10,kernel_size=101,strides=1,padding='same',activation='relu')(input0)
    tensor = MaxPooling1D(11,strides=2)(tensor)
    
    tensor = Conv1D(filters=20,kernel_size=101,strides=1,padding='same',activation='relu')(tensor)
    tensor = MaxPooling1D(11,strides=2)(tensor)
    
    tensor = Conv1D(filters=40,kernel_size=101,strides=1,padding='same',activation='relu')(tensor)
    tensor = MaxPooling1D(11,strides=2)(tensor)
    
    tensor = Flatten()(tensor)  
    tensor = Dense(100,name='FC0')(tensor)
    prediction = Dense(2, activation='softmax',name='output')(tensor)
    model = Model(inputs = input0,outputs = prediction)
 
    # 加载模型权重
    model.load_weights(r'.\model\CNN_3channels\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.h5py')
    # 预测
    prob_trainY = model.predict(X_train)
    prob_testY = model.predict(X_test)
    
    pred_trainY = np.argmax(prob_trainY,axis=1)
    pred_testY = np.argmax(prob_testY,axis=1)    
           
    # 保存重要数据
    important_data = {'pred_trainY':pred_trainY,'pred_testY':pred_testY}
    data_path = r'.\important_data\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
    with open(data_path, 'wb') as fw:
        pickle.dump(important_data, fw)    
#显示结果
Se = []; Sp = []; Ber = []; Auc = []
print('baseline three :CNNE3 ')
print('程序运行 %.1f 分钟'%((time.time()-time_start)/60))
    