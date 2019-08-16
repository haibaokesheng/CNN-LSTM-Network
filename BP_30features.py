# -*- coding: utf-8 -*-
"""
心电传统30特征,BP神经网络
2019年7月6日
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.chdir('D:\project\python_code\ECG_project_2019')
import time;
time_start = time.time();

import pickle
import numpy as np
import scipy.io as sio 
#import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense,Input
from my_functions import calc_ber,calc_auc,auc
from my_loss import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
std = StandardScaler()

length_list = [2,4,6,8]
Fs = 250

units_num = 12   # 如何自动超参数选择？
path = 'BP_30features\\units_num='+str(units_num)
for i in  ['model','loss_curve','important_data']:
    if not os.path.exists(i+'\\'+path):
        os.makedirs(i+'\\'+path)

cv = np.eye(5) #1是测试集  0是训练集
for signal_len in length_list:
    print(signal_len)
    # 加载 30 心电特征
    data = sio.loadmat(r'.\ECG_data\thirty_features_with_annotion\ECG_'+str(signal_len)+'s_5fold_30features_1.mat')['data_ecg']
        
    for k in range(len(cv)):   #交叉验证
        # 训练集索引
        index = np.argwhere(cv[k,]==0).reshape(-1)
        X_train = np.vstack(data[index,0]) 
        Y_train = np.vstack(data[index,1]).reshape(-1) 
        # 测试集索引
        index = np.argwhere(cv[k,]==1).reshape(-1)
        X_test = np.vstack(data[index,0])
        Y_test = np.vstack(data[index,1]).reshape(-1)
        # 打乱测试集
        permutation = np.random.permutation(X_test.shape[0])
        X_test = X_test[permutation,:]
        Y_test = Y_test[permutation]
        # 特征标准化
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
        # 构建神经网络
        input0 = Input(shape=(X_train.shape[1],),name='input0')
        tensor = Dense(units_num,activation='tanh',name='FC')(input0)
        prediction = Dense(1,activation='sigmoid',use_bias='False',name='output')(tensor)
        model = Model(inputs=input0,outputs=prediction) 
        # 训练模型
        model.compile(loss = roc_auc_score ,optimizer ='adam',metrics=[auc])
        history = model.fit(X_train,Y_train,batch_size=256,epochs=50,validation_data=(X_test,Y_test))
        # 保存模型
        model.save('.\model\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.hdf5')
        # 预测
        prob_trainY = model.predict(X_train)
        prob_testY = model.predict(X_test)
        # ROC曲线 tpr = Se,fpr = 1-Sp Youden_index = Se+Sp-1 = tpr-fpr(约登指数)
        fpr,tpr,thresholds = roc_curve(Y_train,prob_trainY,1,drop_intermediate=False)
        Youden_index = tpr-fpr
        # 概率转换成预测标签
        thr = thresholds[np.where(Youden_index == np.max(Youden_index))]
        pred_trainY = (prob_trainY >= thr).astype('int8')
        pred_testY = (prob_testY >= thr).astype('int8')
        # 评价指标
        Se_tr,Sp_tr,Ber_tr = calc_ber(Y_train,pred_trainY)  
        Se_te,Sp_te,Ber_te = calc_ber(Y_test,pred_testY)
        Auc_tr = calc_auc(Y_train,prob_trainY,1)
        Auc_te = calc_auc(Y_test,prob_testY,1)                
        # 绘制代价函数曲线
        train_loss = history.history['loss']
        test_loss = history.history['val_loss']

        train_auc = history.history['auc']
        test_auc = history.history['val_auc']
        # 保存重要实验数据
        important_data = {'train_loss':train_loss,'test_loss':test_loss,
                          'Se':Se_te,'Sp':Sp_te,'Auc':Auc_te,'Ber':Ber_te,
                          'Se_tr':Se_tr,'Sp_tr':Sp_tr,'Auc_tr':Auc_tr,'Ber_tr':Ber_tr,
                          'prob_trainY':prob_trainY,
                          'prob_testY': prob_testY,
                          'threshold':thr,'train_auc':train_auc,'test_auc':test_auc}
        data_path = '.\important_data\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
        with open(data_path, 'wb') as fw:
            pickle.dump(important_data, fw)
#显示结果
for signal_len in length_list:
    Se = []; Sp = []; Ber = []; Auc = []
    for k in range(5):
        data_path = '.\important_data\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
        with open(data_path, 'rb') as f:
            a = pickle.load(f)
            Se.append(a['Se']);Sp.append(a['Sp']);Ber.append(a['Ber']);Auc.append(a['Auc'])        
    print('%ds ecg mean_Se:%.2f%%'%(signal_len,np.mean(Se)))
    print('%ds ecg mean_Sp:%.2f%%'%(signal_len,np.mean(Sp)))
    print('%ds ecg mean_Ber:%.2f%%'%(signal_len,np.mean(Ber)))
    print('%ds ecg mean_Auc:%.3f'%(signal_len,np.mean(Auc)))
    print('\n',end="")

print('30features+BP,units=%d'%(units_num))
print('程序运行 %.1f 分钟'%((time.time()-time_start)/60))
