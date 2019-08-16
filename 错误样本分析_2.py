# -*- coding: utf-8 -*-
"""
可除颤心律信号识别
卷积+顺序池化+LSTM网络
2019年3月25日
错误样本分析
"""
import pickle
import numpy as np
import scipy.io as sio  
import matplotlib.pyplot as plt
from my_functions import calc_ber

from keras.layers import Dense,Input,Lambda,Conv2D,LSTM,concatenate
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from my_functions import slide_window
# 重要模型参数
signal_len = 2     # 信号长度 (s)
window  = 0.6      # 滑动窗宽(s)
step = 0.2         # 滑动窗移动步长      
Fs = 250           # 信号采样率
Ber = []
Error = []
path = 'CNN_2kernels'
# 加载数据
data = sio.loadmat(r'.\ECG_data\public_ECG_5fold_data_with_annotion.mat')['data']

annotation = set(np.vstack(data[:,3]).reshape(-1))
original_label = np.vstack(data[:,3]).reshape(-1)
cv = np.eye(5) #1是测试集  0是训练集
for k in range(len(cv)):   
    # 测试集索引
    index = np.argwhere(cv[k,]==1).reshape(-1)
    ecg = np.vstack(data[index,0])
    Y_test = np.vstack(data[index,1]).reshape(-1)
   # original_label = np.vstack(data[index,3]).reshape(-1)
    # 取前 signal_len 的信号
    ecg = ecg[:,0:int(signal_len*Fs)]
    # 获取滑动窗数据
    X_test = slide_window(ecg,window,step)[0]
    # 增加一维度以适应二维卷积的输入格式
    X_test = np.expand_dims(X_test,axis=-1)
    # 构建模型
    input0 = Input(shape=(X_test.shape[1],X_test.shape[2],1),name='input0')
    conv2d = Conv2D(filters=2,kernel_size=[1,4],strides=1, padding='valid',use_bias=0,name='conv2d',trainable=True)
    tensor = conv2d(input0)
    tensor = Lambda(lambda x:tf.transpose(x,[0,1,3,2]))(tensor)
    # 设定卷积网络的初值
    init_cnn_weight = []
    init_cnn_kernel = np.array([[   [[-1,-1]],[[1,1]],[[0,0]],[[0,0]]  ]])
    init_cnn_weight.append(init_cnn_kernel)
    conv2d.set_weights(init_cnn_weight)
        
    tensor = Lambda(lambda x:tf.nn.top_k(x,x.shape[3])[0],name='sequential_pooling_drop')(tensor) # 在第2个维度上对数据降序排列,即各滑动窗内数据
    tensor = Lambda(lambda x:K.reverse(x, axes=3),name='sequential_pooling_asend')(tensor) # 升序排列

    for i in range(2):
        locals()['tensor'+str(i)] = Lambda(lambda x:x[:,:,i,:])(tensor)
        locals()['tensor'+str(i)] = LSTM(units = 40,name='lstm_'+str(i)) (locals()['tensor'+str(i)])
        locals()['tensor'+str(i)] = Dense(20,activation ='tanh',name='FC'+str(i)) (locals()['tensor'+str(i)]) 
        locals()['prediction'+str(i)] = Dense(1, activation='sigmoid',name='output_'+str(i),use_bias='False') (locals()['tensor'+str(i)])   
    
    tensor = concatenate([prediction0,prediction1])
    prediction = Dense(1, activation='sigmoid',name='final_output',use_bias='False')(tensor)
    model = Model(inputs = input0,outputs = prediction)
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
    Ber.append(calc_ber(Y_test,pred_testY)[2])
    # 找出错误样本的索引
    error_index = np.argwhere((pred_testY-Y_test)!=0).reshape(-1)
    Error.append(error_index)
    print(k+1)
print('%.2f'%np.mean(Ber))  
error = np.hstack([Error[0],Error[1],Error[2],Error[3],Error[4]])
for i in annotation:
    one = np.argwhere(original_label==i).reshape(-1) # 423 AF index
    # 错误样本索引与当前心律的索引的交集
    c = list(set(error).intersection(set(one)))
    acc = (len(one)-len(c))/len(one)
    print('%.2f'%acc)
  