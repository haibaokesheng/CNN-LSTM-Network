# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:23:34 2019

@author: haibao
模型拓扑结构
"""
from keras.layers import Dense,Input,Lambda,Conv2D,LSTM,concatenate
from keras.models import Model
from keras import backend as K
import tensorflow as tf

def my_model_8s(X_test):
    input0 = Input(shape=(X_test.shape[1],X_test.shape[2],1),name='input0')
    conv2d = Conv2D(filters=1,kernel_size=[1,4],strides=1, padding='valid',use_bias=0,name='conv2d',trainable=True)
    tensor = conv2d(input0)  
    tensor = Lambda(lambda x:K.squeeze(x,axis=-1),name='squeeze_0')(tensor) # 4D 数据压缩为 3D 数据,因为channels=1
    tensor = Lambda(lambda x:tf.nn.top_k(x,x.shape[2])[0],name='sequential_pooling_drop')(tensor) # 在第2个维度上对数据降序排列,即各滑动窗内数据
    tensor = Lambda(lambda x:K.reverse(x, axes=2),name='sequential_pooling_asend')(tensor) # 升序排列
    tensor = LSTM(units = 50,name='lstm')(tensor) # 多对一LSTM
    tensor = Dense(20,activation ='tanh',name='FC0')(tensor)
    prediction = Dense(1, activation='sigmoid',name='output',use_bias='False')(tensor)
    model = Model(inputs = input0,outputs = prediction)
    return model

def my_model_4s(X_test):
    input0 = Input(shape=(X_test.shape[1],X_test.shape[2],1),name='input0')
    conv2d = Conv2D(filters=2,kernel_size=[1,4],strides=1, padding='valid',use_bias=0,name='conv2d',trainable=True)
    tensor = conv2d(input0)
    tensor = Lambda(lambda x:tf.transpose(x,[0,1,3,2]))(tensor)
    tensor = Lambda(lambda x:tf.nn.top_k(x,x.shape[3])[0],name='sequential_pooling_drop')(tensor) # 在第2个维度上对数据降序排列,即各滑动窗内数据
    tensor = Lambda(lambda x:K.reverse(x, axes=3),name='sequential_pooling_asend')(tensor) # 升序排列
    
    tensor0 = Lambda(lambda x:x[:,:,0,:])(tensor)
    tensor0 = LSTM(units = 20,name='lstm_'+str(0))(tensor0)
    tensor0 = Dense(20,activation ='tanh',name='FC'+str(0)) (tensor0)
    prediction0 = Dense(1, activation='sigmoid',name='output_'+str(0),use_bias='False') (tensor0)  
    
    
    tensor1 = Lambda(lambda x:x[:,:,1,:])(tensor)
    tensor1 = LSTM(units = 20,name='lstm_'+str(1))(tensor1)
    tensor1 = Dense(20,activation ='tanh',name='FC'+str(1)) (tensor1)
    prediction1 = Dense(1, activation='sigmoid',name='output_'+str(1),use_bias='False') (tensor1)  
    
    tensor = concatenate([prediction0,prediction1])
    prediction = Dense(1, activation='sigmoid',name='final_output',use_bias='False')(tensor)
    model = Model(inputs = input0,outputs = prediction)
    return model

