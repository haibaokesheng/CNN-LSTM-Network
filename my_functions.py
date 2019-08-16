# -*- coding: utf-8 -*-
import numpy as np
from keras import backend as K
"""
auc计算
"""
def calc_auc(true,probability,pos_label=1):
   #  true = true.reshape(-1)
     auc = 0
     pos_prob = np.take(probability,np.argwhere(true == pos_label))#正样本的概率
     neg_prob = np.take(probability,np.argwhere(true!= pos_label))#负样本的概率
     for i in pos_prob:
         for j in neg_prob:
             if i>j:
                 auc += 1
             elif i==j:
                 auc += 0.5
     return auc/(len(pos_prob)*len(neg_prob))
    
"""
Se Sp ber 计算
"""
#y_true = np.array([1,0,1,1,0,0,0]) #1正样本、0负样本
#y_pred = np.array([0,0,1,1,0,1,0])

def calc_ber(y_true,y_pred):
    TP = 0
    TN = 0
#    acc = 0
    P = sum(y_true)     #真实值正样本的个数
    N = len(y_true)-P   #真实值负样本的个数
    for i,j in zip(y_true,y_pred):
        if i==1 and j==1:
            TP = TP+1
        if i==0 and j==0:
            TN = TN+1
#        if i == j:
#            acc = acc+1
    Se = TP/P
    Sp = TN/N
    ber = 1-0.5*(Se+Sp)
#    return TP,TN,P-TP,N-TN
#    acc = acc/(P+N)
    return Se*100,Sp*100,ber*100    
'''
正确率计算
'''
def calc_acc(y_true,y_pred):
    acc = 0
    for i,j in zip(y_true,y_pred):
        if i == j:
            acc = acc+1
    return (acc/(len(y_true)))*100

'''
得到滑动窗数据
'''
def slide_window(data,window,step,Fs=250):  
    timestep = round(((data.shape[1]/Fs)-window)/step)+1 #滑动窗数量
    for i in range(0,timestep):
        if i==0:
            X = data[:,round((i*step)*Fs):round((i*step+window)*Fs)]
        else:
            b = data[:,round(i*step*Fs):round((i*step+window)*Fs)]    
            X = np.hstack((X,b))
    return X.reshape(-1,timestep,int(window*Fs)),timestep

"""
平滑曲线
last : 上一数据点
point :当前数据点
weight :平滑系数
"""
def smooth_value(last,point,weight=0.85):
      smoothed_val =  last* weight + (1 - weight) * point
      return smoothed_val

'''
(4)ber评价指标
''' 
def ber(y_true, y_pred):
    threshold = K.variable(value=0.5)
    #Se
    y_pred = K.cast(y_pred >= threshold, 'float32')  
    P = K.cast(K.sum(y_true),'float32')  
    TP = K.sum(y_pred * y_true)
    Se = TP/P
    #Sp
    N = K.cast(K.sum(1 - y_true),'float32')
    FP = K.sum(y_pred - y_pred * y_true)
    FPR = FP/N
    Sp = 1-FPR
    
    return 100*(1-0.5*(Se+Sp))
"""
记录ber
"""
import keras
class BerHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.ber = {'batch':[], 'epoch':[]}
        self.val_ber = {'batch':[], 'epoch':[]}
        self.loss = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
    
    def on_batch_end(self, batch, logs={}):

        self.ber['batch'].append(logs.get('ber'))
        self.val_ber['batch'].append(logs.get('val_ber'))
        self.loss['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        
    def on_epoch_end(self, batch, logs={}):
        self.ber['epoch'].append(logs.get('ber'))
        self.val_ber['epoch'].append(logs.get('val_ber'))
        self.loss['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

'''
平滑约束正则项
'''
import tensorflow as tf
#import numpy as np
#from keras import backend as K
#weight_matrix = np.random.randn(147,8)
def neighbor_reg(weight_matrix):
    W1 = tf.gather(weight_matrix,np.arange(0,146))
    #print(W1)
    W2 = tf.gather(weight_matrix,np.arange(1,147))
    #print(W2)
    return 0.02 * K.sum(K.square(W1-W2))

#print(neighbor_reg(weight_matrix))

#print(K.eval(neighbor_reg(weight_matrix)))
           

def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
