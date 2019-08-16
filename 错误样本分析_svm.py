# -*- coding: utf-8 -*-
"""

"""
import pickle
import numpy as np
import scipy.io as sio  
from sklearn.preprocessing import StandardScaler
from my_functions import calc_ber


signal_len = 8
std = StandardScaler()
Error = []
Ber = []
# 加载数据
data = sio.loadmat(r'.\ECG_data\30features_plos_one\ECG_'+str(signal_len)
                        +'s_5fold_30features_1.mat')['data_ecg']
original_label = np.vstack(data[:,3]).reshape(-1)
annotation = [21,19,20,2,5,6,8,10,11,13,15,16,20] # 论文中的顺序
name = ['VT','VF','VFL','AFIB','B','BI','HGEA','NSR','NOD','PM','SBR','SVTA','VER']   

cv = np.eye(5) #1是测试集  0是训练集
for k in range(len(cv)): #5重
    print(k)
    index = np.argwhere(cv[k,]==0).reshape(-1)
    X_train = np.vstack(data[index,0])               # 加载训练数据是为了得到均值和方差的           
    # 测试集索引
    index = np.argwhere(cv[k,]==1).reshape(-1)
    X_test = np.vstack(data[index,0])
    Y_test = np.vstack(data[index,1]).reshape(-1)
    patient_id_test = np.vstack(data[index,2]).reshape(-1)
    original_label_test  = np.vstack(data[index,3]).reshape(-1) 
    # 特征标准化
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    #X_test = std.fit_transform(X_test)
    # 加载 svm 模型
    file_path = r'.\model\svm_model_30features\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
    with open(file_path, 'rb') as fr:
        new_svm = pickle.load(fr)
    # 预测
    pred_testY = new_svm.predict(X_test)
    Ber.append(calc_ber(Y_test,pred_testY)[2])

    # 找出错误样本的索引
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
print('Ber:%.2f%%'%np.mean(Ber)) 

import pandas as pd
df = pd.DataFrame(Acc,index =name)  
df = df.transpose()

    