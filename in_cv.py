# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:31:35 2018

@author: Administrator
"""
import math
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler
std = StandardScaler()

def in_cv_30featurs(k,number,patient_id,X,Y,g_range,C_range,in_fold=5):
    max_ber = 1.0 #ber理论最大值
    for gamma in g_range:#核半径
        for C in C_range:

#    for gamma in [5e-5,1e-4,1e-3,1e-2,1e-1,1]:#核半径
#        for C in [0.01,0.1,1,10,100,1000]:

            Se_list_in = []
            Sp_list_in = []
            ber_list_in = []
            svm = SVC(gamma=gamma,C=C)
            # 内层交叉验证(按照病人划分样本)
            #permutation = np.random.permutation(number)
            fold_list = [math.ceil(len(number)/in_fold)]*4
            fold_list.append(len(number)-sum(fold_list))#每一重的病人的数量
            # 生成交叉验证索引列表
            cv_list = []
            f = []
            for i in range(in_fold):
                f.append(sum(fold_list[0:i]))
            for i in range(in_fold):
                if(i<in_fold-1):
                    cv_list.append(number[f[i]:f[i+1]])
                else:
                    cv_list.append(number[f[i]:])
            # 划分数据集
            for i in range(in_fold):#内部5重交叉验证
                print('外第 %d 重CV,g = %.1e,C = %.1e,内第 %d 重CV'%(k+1,gamma,C,(i+1)))
                test_index = []
                train_index = []
                
                for find in (cv_list[i]):
                    tem = [n for n,v in enumerate(patient_id) if v==find]
                    test_index.extend(tem)
                train_index_num = list(set(number).difference(set(cv_list[i])))  # b中有而a中没有的  
                for find in (train_index_num):
                    tem = [n for n,v in enumerate(patient_id) if v==find]
                    train_index.extend(tem)
                    
                train_X = X[train_index,:]
                train_Y = Y[train_index]
                
                test_X = X[test_index,:]
                test_Y = Y[test_index]
                # 特征标准化
                train_X_std = std.fit_transform(train_X)
                test_X_std = std.transform(test_X)
                # 训练模型
                svm.fit(train_X_std,train_Y)
                # 预测
                pred_testY = svm.predict(test_X_std)
                # 评价指标
                Se_test_in = recall_score(y_true=test_Y,y_pred=pred_testY)
                confmat_test = confusion_matrix(y_true=test_Y,y_pred=pred_testY)
                Sp_test_in = (confmat_test[0,0]/(confmat_test[0,0]+confmat_test[0,1]))
                ber = 1-0.5*(Se_test_in+Sp_test_in)
                Se_list_in.append(Se_test_in)
                Sp_list_in.append(Sp_test_in)
                ber_list_in.append(ber)
    
            mean_ber = np.mean(ber_list_in) #取均值
            if  mean_ber < max_ber:
                max_ber = mean_ber
                best_parameters = {"gamma":gamma,"C":C}
                print('Ber:%.2f%%'%(mean_ber*100))
            else:
                print('\n',end="")
    return best_parameters


def in_cv_sos(k,number,patient_id,X,Y,g_range,C_range,in_fold=5):
    max_ber = 1.0 #ber理论最大值
    for gamma in g_range:#核半径
        for C in C_range:
#    for gamma in [5e-5,1e-4,1e-3,1e-2,1e-1,1]:#核半径
#        for C in [0.01,0.1,1,10,100,1000,5000,1e4]:

            Se_list_in = []
            Sp_list_in = []
            ber_list_in = []
            svm = SVC(gamma=gamma,C=C)
            # 内层交叉验证(按照病人划分样本)
            #permutation = np.random.permutation(number)
            fold_list = [math.ceil(len(number)/in_fold)]*4
            fold_list.append(len(number)-sum(fold_list))#每一重的病人的数量
            # 生成交叉验证索引列表
            cv_list = []
            f = []
            for i in range(in_fold):
                f.append(sum(fold_list[0:i]))
            for i in range(in_fold):
                if(i<in_fold-1):
                    cv_list.append(number[f[i]:f[i+1]])
                else:
                    cv_list.append(number[f[i]:])
            # 划分数据集
            for i in range(in_fold):#内部5重交叉验证
                print('外第 %d 重CV,g = %.1e,C = %.1e,内第 %d 重CV'%(k+1,gamma,C,(i+1)))
                test_index = []
                train_index = []
                
                for find in (cv_list[i]):
                    tem = [n for n,v in enumerate(patient_id) if v==find]
                    test_index.extend(tem)
                train_index_num = list(set(number).difference(set(cv_list[i])))  # b中有而a中没有的  
                for find in (train_index_num):
                    tem = [n for n,v in enumerate(patient_id) if v==find]
                    train_index.extend(tem)
                    
                train_X = X[train_index,:]
                train_Y = Y[train_index]
                
                test_X = X[test_index,:]
                test_Y = Y[test_index]
            
                # 训练模型
                svm.fit(train_X,train_Y)
                # 预测
                pred_testY = svm.predict(test_X)
                # 评价指标
                Se_test_in = recall_score(y_true = test_Y,y_pred = pred_testY)
                confmat_test = confusion_matrix(y_true = test_Y,y_pred = pred_testY)
                Sp_test_in = (confmat_test[0,0]/(confmat_test[0,0]+confmat_test[0,1]))
                ber = 1-0.5*(Se_test_in+Sp_test_in)
                Se_list_in.append(Se_test_in)
                Sp_list_in.append(Sp_test_in)
                ber_list_in.append(ber)
    
            mean_ber = np.mean(ber_list_in) #取均值
            if  mean_ber < max_ber:
                max_ber = mean_ber
                best_parameters = {"gamma":gamma,"C":C}
                print('Ber:%.2f%%'%(mean_ber*100))
            else:
                print('\n',end="")
    return best_parameters,max_ber


#if __name__ == '__main__':#当.py文件以模块形式被导入时,之下的代码块不被运行。
#    in_cv(k,number,patient_id,X,Y)