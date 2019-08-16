# -*- coding: utf-8 -*-
"""
斜率顺序统计量特征+支持向量机分类
最新修改于2019年7月5日
"""
import time;
time_start = time.time();
import numpy as np
import pickle
import scipy.io as sio  
from sklearn.svm import SVC
from in_cv import in_cv_sos
from my_functions import calc_ber

Fs = 250
signal_len = 2

cv_flag = 1
cv = np.eye(5) #1是测试集  0是训练集
# 超参数列表
if signal_len == 2:
    g_range = [5e-5,2.5e-5,1e-5,5e-5,1e-4,1e-3,1e-2]    # g越大,支持向量个数越少,高斯分布越高越瘦
    C_range = [7.5e5,1e6,1e7,1e8,1e9,2e9,2.5e9,3e9]  # C越大,每个样本都要分对,易过拟合
elif signal_len == 4:
    g_range = [1e-5,1e-4,1e-3,1e-2,1e-1]
    C_range = [1e2,1e3,1e4,5e4,1e5,5e5,1e6] 
elif signal_len == 6:
    g_range = [1e-5,1e-4,1e-3,1e-2,1e-1,1] 
    C_range = [1e2,1e3,1e4,5e4,1e5,5e5,1e6,1e7]
elif signal_len == 8:
    g_range = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1] 
    C_range = [1e2,1e3,1e4,5e4,1e5,5e5,1e6,1e7]
# 加载数据
data = sio.loadmat(r'.\ECG_data\public_ECG_5fold_data_with_annotion_1.mat')['data']
Se_list = []
Sp_list = []
ber_list = []
ber_in = []
for k in range(len(cv)):
#k = 4   
# 训练集
    index = np.argwhere(cv[k,]==0).reshape(-1)
    X_train = np.vstack(data[index,0])  
    Y_train = np.vstack(data[index,1]).reshape(-1) 
    patient_id_train = np.vstack(data[index,2]).reshape(-1)#训练集受试者编号
    number = list(set(patient_id_train))
    # 测试集
    index = np.argwhere(cv[k,]==1).reshape(-1)
    X_test = np.vstack(data[index,0])
    Y_test = np.vstack(data[index,1]).reshape(-1)
    # 取前 signal_len 秒的 ecg 信号
    X_train =  X_train[:,0:int(signal_len*Fs)]
    X_test = X_test[:,0:int(signal_len*Fs)]
    
    print('%d fold 训练集 ShR:%d,NshR:%d'%((k+1),len(np.argwhere(Y_train==1)),len(np.argwhere(Y_train==0))))
    print('%d fold 测试集 ShR:%d,NshR:%d'%((k+1),len(np.argwhere(Y_test==1)),len(np.argwhere(Y_test==0))))
    # 差分
    X_train = np.diff(X_train)
    X_test = np.diff(X_test)
    # 排序
    X_train = np.sort(X_train)
    X_test = np.sort(X_test)
    # 嵌套交叉验证确定超参
    if cv_flag:
        best_parameters,b = in_cv_sos(k,number,patient_id_train,X_train,Y_train,g_range,C_range)
        ber_in.append(b)
        svm = SVC(**best_parameters)
    else:
        svm = SVC(gamma=0.1,C=10)
    # 训练模型
    svm.fit(X_train,Y_train)
    # 保存模型
    file_path = '.\model\svm_model_sos\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
    with open(file_path, 'wb') as fw:
        pickle.dump(svm, fw)
    # 预测
    pred_testY = svm.predict(X_test)
    # 评价指标
    Se_test,Sp_test,ber = calc_ber(Y_test,pred_testY)
    Se_list.append(Se_test)
    Sp_list.append(Sp_test)
    ber_list.append(ber)

# 保存实验结果
result = np.vstack([Se_list,Sp_list,ber_list])
np.save(r'important_data\svm_sos_result\svm_'+str(signal_len)+'s',result)

result = np.load(r'important_data\svm_sos_result\svm_'+str(signal_len)+'s.npy')
#result[0,4] = Se_test
#result[1,4] = Sp_test
#result[2,4] = ber

print('差分排序+svm,信号%ds'%(signal_len))
print('mean_Se:%.2f%%'%(np.mean(result[0,:])))
print('mean_Sp:%.2f%%'%(np.mean(result[1,:])))
print('mean_ber:%.2f%%\n'%(np.mean(result[2,:])))
print('程序运行 %.1f 分钟'%((time.time()-time_start)/60))
# 检查模型超参数
for k in range(5):
    file_path = '.\model\svm_model_sos\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
    with open(file_path, 'rb') as fr:
        new_svm = pickle.load(fr)
       # print ('%ds 第 %d 重 g = %.1e,C = %.1e'%(signal_len,(k+1),new_svm.gamma,new_svm.C))

        print ('%ds 第 %d 重 g = %.1e,C = %.1e,in_ber=%.2f%%'%(signal_len,(k+1),new_svm.gamma,new_svm.C,ber_in[k]*100))
print('\n',end="")