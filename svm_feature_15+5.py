# -*- coding: utf-8 -*-
"""
15特征 + 5种VMED 特征 
分类器：SVM 
2019.8.11
"""
import time;
import os

time_start = time.time();
import numpy as np
import pandas as pd
import pickle
import scipy.io as sio  
from sklearn.svm import SVC
from in_cv import in_cv_30featurs
from my_functions import calc_ber
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
cv_flag = 1
cv = np.eye(5) #1是测试集  0是训练集
path = 'svm_20features'
for i in  ['model','important_data']:
    if not os.path.exists(i+'\\'+path):
        os.makedirs(i+'\\'+path)

# 传统特征 id
name = ['tci','tcsc','exp','expmod','cm','cvbin',
        'frqbin','abin','kurt','vfleak','M','A1','A2','A3',
        'mav','psr','hilb','SamEn','x1','x2','x3','x4','x5',
        'bCP','bWT','BW','Li','count1','count2','count3','CF','CP','PSA'] # 33 种特征
# 15种特征(CFS3)
name_1 =['vfleak','count2','bCP','bWT','cm','count1','exp','tcsc','BW','psr','PSA','count3','CF','SamEn','frqbin'] 
# 5种 VMED 特征
name_VMED = ['vfleak','bWT','mav','count2','tcsc']

signal_len = 2
# 超参数列表  选择原则,最佳参数不要出现在边界
if signal_len == 2:
    g_range = [5e-5,1e-4,1e-3,1e-2,1e-1,1]   # g越大,支持向量个数越少,高斯分布越高越瘦
    C_range = [0.01,0.1,1,10,100,1000]       # C越大,每个样本都要分对,易过拟合
elif signal_len == 4:
    g_range = [5e-5,1e-4,1e-3,1e-2,1e-1,1]
    C_range = [0.01,0.1,1,10,1e2,1e3,1e4] 
elif signal_len == 6:
    g_range = [5e-5,1e-4,1e-3,1e-2,1e-1,1]
    C_range = [0.01,0.1,1,10,100,1000] 
elif signal_len == 8:
    g_range = [0.0001,0.001,0.005,0.01,0.02,0.03,0.04,0.1]
    C_range = [5,10,15,20,25,30,35,40] 
# 加载 33 特征的数据
data = sio.loadmat(r'.\ECG_data\thirty_three_features_with_annotion\ECG_'+str(signal_len)+'s_5fold_33features_1.mat')['data_ecg']
# 加载 VMED 特征数据(也是33个)
data_vmed = sio.loadmat(r'.\ECG_data\VMED_features\ECG_'+str(signal_len)+'s_5fold_33features_vmed.mat')['data_ecg']

Se_list = []
Sp_list = []
ber_list = []

for k in range(len(cv)): #5重
    #print(k)
    # 训练集
    index = np.argwhere(cv[k,]==0).reshape(-1)
    X_train = np.vstack(data[index,0])                           # 第 0 列是传统特征
    Y_train = np.vstack(data[index,1]).reshape(-1)               # 第 1 列是0-1的标记
    patient_id_train = np.vstack(data[index,2]).reshape(-1)      # 第 2 列是受试者编号
    number = list(set(patient_id_train))
    X_train = pd.DataFrame(X_train)
    X_train.columns = name
    X_train = pd.DataFrame(X_train,columns = name_1)
    
    VMED_train = np.vstack(data_vmed[index,0])
    VMED_train = pd.DataFrame(VMED_train,columns = name)
    VMED_train = pd.DataFrame(VMED_train,columns = name_VMED)
    X_train = pd.concat([X_train, VMED_train],axis=1)
    # 测试集
    index = np.argwhere(cv[k,]==1).reshape(-1)
    X_test = np.vstack(data[index,0])
    Y_test = np.vstack(data[index,1]).reshape(-1)   
    X_test = pd.DataFrame(X_test)
    X_test.columns = name
    X_test = pd.DataFrame(X_test,columns = name_1)
    
    VMED_test = np.vstack(data_vmed[index,0])
    VMED_test = pd.DataFrame(VMED_test,columns = name)
    VMED_test = pd.DataFrame(VMED_test,columns = name_VMED)
    X_test = pd.concat([X_test, VMED_test],axis=1)

    # 嵌套交叉验证确定超参
    if cv_flag:
        X_train = np.array(X_train)
        best_parameters = in_cv_30featurs(k,number,patient_id_train,X_train,Y_train,g_range,C_range)
        svm = SVC(**best_parameters)
    else:
        svm = SVC(gamma=0.04,C=10)
      #  svm = SVC()
    # 特征标准化
    X_train_std = std.fit_transform(X_train)
    X_test_std = std.transform(X_test)
    # 训练模型
    svm.fit(X_train_std,Y_train)
    # 保存模型
    file_path = r'.\model\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
    with open(file_path, 'wb') as fw:
        pickle.dump(svm, fw)
    # 预测
    pred_testY = svm.predict(X_test_std)
  
    Se_test,Sp_test,ber = calc_ber(Y_test,pred_testY)
    Se_list.append(Se_test)
    Sp_list.append(Sp_test)
    ber_list.append(ber)
    
#保存实验结果
result = np.vstack([Se_list,Sp_list,ber_list])
np.save(r'important_data\\'+path+'\svm_'+str(signal_len)+'s',result)

# 读取实验结果
result = np.load(r'important_data\\'+path+'\svm_'+str(signal_len)+'s.npy')
print('baseline 2 :15features+5vmed svm,信号%ds'%(signal_len))
print('mean_Se:%.2f%%'%(np.mean(result[0,:])))
print('mean_Sp:%.2f%%'%(np.mean(result[1,:])))
print('mean_ber:%.2f%%\n'%(np.mean(result[2,:])))

print('程序运行 %.1f 分钟'%((time.time()-time_start)/60))

# 检查模型超参数
for k in range(5):
    file_path = r'.\model\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
    with open(file_path, 'rb') as fr:
        new_svm = pickle.load(fr)
        print ('%ds 第 %d 重 g = %.4f,C = %d'%(signal_len,(k+1),new_svm.gamma,new_svm.C))
print('\n',end="")
