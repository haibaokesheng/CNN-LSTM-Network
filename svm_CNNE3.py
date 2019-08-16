# -*- coding: utf-8 -*-
"""
CNNE3网络的输出作为特征， svm 分类
最新修改于2019年8月11日
"""
import time;
time_start = time.time();
import numpy as np

import pickle
import scipy.io as sio  
from sklearn.svm import SVC
from in_cv import in_cv_30featurs
from my_functions import calc_ber
#from sklearn.preprocessing import StandardScaler
#std = StandardScaler()
cv_flag = 0
cv = np.eye(5) #1是测试集  0是训练集

signal_len = 2
# 超参数列表
if signal_len == 2:
    g_range = [5e-5,1e-4,1e-3,1e-2,1e-1,1]   # g越大,支持向量个数越少,高斯分布越高越瘦
    C_range = [0.01,0.1,1,10,100,1000]  # C越大,每个样本都要分对,易过拟合
elif signal_len == 4:
    g_range = [5e-5,1e-4,1e-3,1e-2,1e-1,1]
    C_range = [0.01,0.1,1,10,1e2,1e3,1e4] 
elif signal_len == 6:
    g_range = [5e-5,1e-4,1e-3,1e-2,1e-1,1]
    C_range = [0.01,0.1,1,10,100,1000] 
elif signal_len == 8:
    g_range = [1e-6,5e-5,1e-4,1e-3,1e-2,1e-1,1]
    C_range = [0.01,0.1,1,10,100,1000] 

# 加载label
data = sio.loadmat(r'.\ECG_data\thirty_features_with_annotion\ECG_'+str(signal_len)+'s_5fold_30features_1.mat')['data_ecg']

Se_list = []
Sp_list = []
ber_list = []

for k in range(len(cv)): #5重
    # 加载 CNNE 特征 
    data_path = r'.\important_data\CNNE3\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
    with open(data_path, 'rb') as f:
        CNNE3 = pickle.load(f)
    # 训练集
    index = np.argwhere(cv[k,]==0).reshape(-1)
    X_train = CNNE3['pred_trainY']                      
    Y_train = np.vstack(data[index,1]).reshape(-1)               # 第 1 列是0-1的标记
    patient_id_train = np.vstack(data[index,2]).reshape(-1)      # 第 2 列是受试者编号
    number = list(set(patient_id_train))
    # 测试集
    index = np.argwhere(cv[k,]==1).reshape(-1)
    X_test =  CNNE3['pred_testY']   
    Y_test = np.vstack(data[index,1]).reshape(-1)    
    # 嵌套交叉验证确定超参
    if cv_flag:
        best_parameters = in_cv_30featurs(k,number,patient_id_train,X_train,Y_train,g_range,C_range)
        svm = SVC(**best_parameters)
    else:
        svm = SVC(gamma=0.1,C=10)
    # 训练模型
    svm.fit(X_train,Y_train)
    # 保存模型
    file_path = r'.\model\svm_CNNE3\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
    with open(file_path, 'wb') as fw:
        pickle.dump(svm, fw)
    # 预测
    pred_testY = svm.predict(X_test)
  
    Se_test,Sp_test,ber = calc_ber(Y_test,pred_testY)
    Se_list.append(Se_test)
    Sp_list.append(Sp_test)
    ber_list.append(ber)
    
#保存实验结果
result = np.vstack([Se_list,Sp_list,ber_list])
np.save(r'important_data\svm_CNNE3\svm_'+str(signal_len)+'s',result)

# 读取实验结果
result = np.load(r'important_data\svm_CNNE3\svm_'+str(signal_len)+'s.npy')
print('30features+svm,信号%ds'%(signal_len))
print('mean_Se:%.2f%%'%(np.mean(result[0,:])))
print('mean_Sp:%.2f%%'%(np.mean(result[1,:])))
print('mean_ber:%.2f%%\n'%(np.mean(result[2,:])))

print('程序运行 %.1f 分钟'%((time.time()-time_start)/60))

# 检查模型超参数
for k in range(5):
    file_path = r'.\model\svm_CNNE3\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
    with open(file_path, 'rb') as fr:
        new_svm = pickle.load(fr)
        print ('%ds 第 %d 重 g = %.1e,C = %.1e'%(signal_len,(k+1),new_svm.gamma,new_svm.C))
print('\n',end="")
