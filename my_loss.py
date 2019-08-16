# -*- coding: utf-8 -*-
"""
我的损失函数。
"""
import numpy as np
import tensorflow as tf
#y_true = np.array([1,0,1,1,0,0,0]) #1正样本、0负样本
#y_pred = np.array([0,0,1,1,0,1,0])
#y_scores = np.array([0.43,0.2,0.79,0.8,0.11,0.82,0.19]) #正样本的概率
    
'''
ber损失 8月5日写的
''' 
def ber_loss (y_true,y_pred):
    #正样本的个数
    #print('正样本个数')
    num_pos = tf.to_float(tf.reduce_sum(y_true))
    #print(num_pos.eval())
    
    #print('正样本是正样本的概率')
    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool)) #正样本是正样本的概率
    #print(pos.eval())
  
    #print('都减去0.5')
    pos = tf.to_float(pos-0.5)
    #print(pos.eval())
    
    #print('S函数之后代替指示函数')   
    #pos = tf.div(1.0,1+tf.exp(tf.multiply(-80.0,pos)))
    pos = tf.div(1.0,1+tf.exp(tf.multiply(-60.0,pos)))
    #print(pos.eval())
    #print('求和')
    pos = tf.reduce_sum(pos)
    
    #print('Se')
    Se = tf.div(pos,num_pos)
    #print(Se.eval())
    
    #负样本
    #print('负样本个数')
    num_neg = tf.to_float(tf.size(y_true))-num_pos
    #print(num_neg.eval())
    #print('负样本是正样本的概率')
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool)) #负样本是正样本的概率
    #print(neg.eval())
    
    #print('减0.5的结果')
    neg = tf.to_float(neg-0.5)
    #print(neg.eval())
    
    #print('S函数之后')   
    neg = tf.div(1.0,1+tf.exp(tf.multiply(-80.0,neg)))
    #print(neg.eval())
    
    #print('1减S函数')
    neg = 1.0-neg
    #print(neg.eval())

    #print('求和')
    neg = tf.reduce_sum(neg)
    #print(neg.eval())
    
    #print('Sp')
    Sp = tf.div(neg,num_neg)
    #print(Sp.eval())
    
    return 1-0.5*(Sp+Se)
#with tf.Session() as sess:
#    
#    sess.run(ber_direct_loss (y_pred=y_scores,y_true=y_true))
#    
'''
auc 损失函数
'''
def roc_auc_score(y_true,y_pred):
    """ ROC AUC Score.
    """
    with tf.name_scope("RocAucScore"):

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p     = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

        return tf.reduce_sum(tf.pow(-masked, p))

       
'''
G_mean损失
''' 
def G_mean_loss (y_true,y_pred):
    #正样本的个数
    #print('正样本个数')
    #num_pos = tf.to_float(tf.reduce_sum(y_true))
    #print(num_pos.eval())
    
    #print('正样本是正样本的概率')
    pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool)) #正样本是正样本的概率
    #print(pos.eval())
  
    #print('都减去0.5')
    pos = tf.to_float(pos-0.5)
    #print(pos.eval())
    
    #print('S函数之后')   
    pos = tf.div(1.0,1+tf.exp(tf.multiply(-80.0,pos)))
    #print(pos.eval())
    #print('求和')
    pos = tf.reduce_sum(pos)
    
    #print('Se')
    #Se = tf.div(pos,num_pos)
    #print(Se.eval())
    
    #负样本
    #print('负样本个数')
    #num_neg = tf.to_float(tf.size(y_true))-num_pos
    #print(num_neg.eval())
    #print('负样本是正样本的概率')
    neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool)) #负样本是正样本的概率
    #print(neg.eval())
    
    #print('减0.5的结果')
    neg = tf.to_float(neg-0.5)
    #print(neg.eval())
    
    #print('S函数之后')   
    neg = tf.div(1.0,1+tf.exp(tf.multiply(-80.0,neg)))
    #print(neg.eval())
    
    #print('1减S函数')
    neg = 1.0-neg
    #print(neg.eval())

    #print('求和')
    neg = tf.reduce_sum(neg)
    #print(neg.eval())
    
    #print('Sp')
    #Sp = tf.div(neg,num_neg)
    #print(Sp.eval())
    
    return -pos*neg
#with tf.Session() as sess:
#    
#    sess.run(ber_direct_loss (y_pred=y_scores,y_true=y_true))
#           
  
'''
https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
F1 评价指标和F1 loss
'''
from keras import backend as K
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    #tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    #tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)
'''
修改版
'''
y_true = np.array([[0.,0.,1.,1],[0.,1.,1.,0],[1,0,0,1]])
#y_true = y_true.transpose()
y_pred = np.array([[0.8,0.05,0.97,0.88],[0.01,0.84,0.88,0.19],[0.15,0.98,0.58,0.44]])  # 2个样本
#y_pred = y_pred.transpose()
def f1_loss_my(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    #print(K.cast(y_true*y_pred, 'float').eval())
    print('tp',tp.eval())
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    print('fp',fp.eval())
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    print('fn',fn.eval())
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    print('f1',f1.eval())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    #print(f1.eval())
    return 1 - K.mean(f1)

def f1_my(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    #tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    print('f1',f1.eval())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


#with tf.Session() as sess:
#    sess.run(f1_my(y_true,y_pred))        
#from sklearn.metrics import f1_score
#print(f1_score(y_true=[0,1,0],y_pred=[0,1,1]))