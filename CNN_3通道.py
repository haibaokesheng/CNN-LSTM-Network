# -*- coding: utf-8 -*-
"""
可除颤心律信号识别
CNN 网络 3个通道 
2019.8.5
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
from keras import utils
#from my_loss import roc_auc_score

from my_functions import calc_ber,calc_auc,auc
# 重要模型参数
signal_len = 6     # 信号长度 (s)
Fs = 250           # 信号采样率  
path = 'CNN_1channels'
cv = np.eye(5) 
# 加载原始数据
data = sio.loadmat(r'.\ECG_data\public_ECG_5fold_data_with_annotion_1.mat')['data']
# 加载 1-12 Hz的数据,即VMED 的123
#data_mved_123  = sio.loadmat(r'.\ECG_data\VMED_123\ECG_MVED_'+str(signal_len)+'s_123.mat')['data_ecg_MVED_123']
## 加载 13-17 Hz的数据  VMED 的78
#data_mved_78  = sio.loadmat(r'.\ECG_data\VMED_78\ECG_MVED_'+str(signal_len)+'s_78.mat')['data_ecg_MVED']

for k in range(len(cv)):   #交叉验证
    # 训练集索引
    index = np.argwhere(cv[k,]==0).reshape(-1)
    X_train = np.vstack(data[index,0])             # 第 0 列是原始信号
#    X_train_123 = np.vstack(data_mved_123[index,0])
#    X_train_78 = np.vstack(data_mved_78[index,0])
    Y_train = np.vstack(data[index,1]).reshape(-1) # 第 1 列是标记信息
    # 测试集索引
    index = np.argwhere(cv[k,]==1).reshape(-1)
    X_test = np.vstack(data[index,0])
#    X_test_123 = np.vstack(data_mved_123[index,0])
#    X_test_78 = np.vstack(data_mved_78[index,0])
    Y_test = np.vstack(data[index,1]).reshape(-1)

    # label 转 one-hot 编码  0-->10,1-->01
    Y_train_oh = utils.to_categorical(Y_train, 2)
    Y_test_oh = utils.to_categorical(Y_test, 2)
    
    # 原始数据取前 signal_len 的信号
    X_train = X_train[:,0:int(signal_len*Fs)]
    X_test = X_test[:,0:int(signal_len*Fs)]
    X_train = np.expand_dims(X_train,axis=-1)
    X_test = np.expand_dims(X_test,axis=-1)

    # 信号拼接
#    X_train = np.dstack([X_train,X_train_123,X_train_78])
#    X_test = np.dstack([X_test,X_test_123,X_test_78])
    
#    del X_train_123,X_test_123,X_train_78,X_test_78
    
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
    # 中间输出
    #    layer_name = 'sequential_pooling_asend'
    #    intermediate_layer_model = Model(input=model.input,
    #                                 output=model.get_layer(layer_name).output)
    #    intermediate_output = intermediate_layer_model.predict(X_train)
    #    intermediate_output = np.squeeze(intermediate_output)
    # 训练模型
    model.compile(loss ='categorical_crossentropy',optimizer ='adam',metrics=[auc])
    #model.compile(loss = roc_auc_score ,optimizer ='adam',metrics=[auc])
    history = model.fit(X_train,Y_train_oh,batch_size=256,epochs=20,validation_data=(X_test,Y_test_oh))
    # 保存模型
    model.save(r'.\model\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.h5py')
    # 预测
    prob_trainY = model.predict(X_train)
    prob_testY = model.predict(X_test)
    
    pred_trainY = np.argmax(prob_trainY,axis=1)
    pred_testY = np.argmax(prob_testY,axis=1)
    
    # ROC曲线 tpr = Se,fpr = 1-Sp Youden_index = Se+Sp-1 = tpr-fpr(约登指数)
    #fpr,tpr,thresholds = roc_curve(Y_train,prob_trainY,1,drop_intermediate=False)
    #Youden_index = tpr-fpr
    # 概率转换成预测标签
    #thr = thresholds[np.where(Youden_index == np.max(Youden_index))]
    #pred_trainY = (prob_trainY >= thr).astype('int8')
    #pred_testY = (prob_testY >= thr).astype('int8')
    # 评价指标
    Se_tr,Sp_tr,Ber_tr = calc_ber(Y_train,pred_trainY)  
    Se_te,Sp_te,Ber_te = calc_ber(Y_test,pred_testY)
    
    Auc_tr = calc_auc(Y_train,np.argmax(prob_trainY,axis=1),1)
    Auc_te = calc_auc(Y_test,np.argmax(prob_testY,axis=1),1)
           
    # 绘制代价函数曲线
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']
      
    train_auc = history.history['auc']
    test_auc = history.history['val_auc']
       
    # 保存重要数据
    important_data = {'train_loss':train_loss,'test_loss':test_loss,
                      'Se':Se_te,'Sp':Sp_te,'Auc':Auc_te,'Ber':Ber_te,
                      'Se_tr':Se_tr,'Sp_tr':Sp_tr,'Auc_tr':Auc_tr,'Ber_tr':Ber_tr,
                      'prob_trainY':prob_trainY,
                      'prob_testY': prob_testY,
                      #'threshold':thr,
                      'train_auc':train_auc,'test_auc':test_auc}
    data_path = r'.\important_data\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
    with open(data_path, 'wb') as fw:
        pickle.dump(important_data, fw)    
#显示结果
Se = []; Sp = []; Ber = []; Auc = []
for k in range(5):
    data_path = r'.\important_data\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
    with open(data_path, 'rb') as f:
        a = pickle.load(f)
        Se.append(a['Se']);Sp.append(a['Sp']);Ber.append(a['Ber']);Auc.append(a['Auc'])        
print('baseline three : 3通道CNN网络')
print('%ds ecg mean_Se:%.2f%%'%(signal_len,np.mean(Se)))
print('%ds ecg mean_Sp:%.2f%%'%(signal_len,np.mean(Sp)))
print('%ds ecg mean_Ber:%.2f%%'%(signal_len,np.mean(Ber)))
print('%ds ecg mean_Auc:%.3f'%(signal_len,np.mean(Auc)))
print('\n',end="")
print('程序运行 %.1f 分钟'%((time.time()-time_start)/60))
    