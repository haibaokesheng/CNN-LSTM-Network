# -*- coding: utf-8 -*-
"""
画出中间结果
2019.6.18
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense,Input,Lambda,Conv2D,LSTM
import scipy.io as sio
from my_functions import slide_window
from keras import backend as K
import tensorflow as tf

# 模型参数
signal_len = 2                    
Fs = 250                          
window = 0.6                                 
step = 0.2      
               
# 加载数据
data = sio.loadmat(r'.\ECG_data\public_ECG_5fold_data_with_annotion')['data']
X = np.vstack(data[:,0])
Y = np.vstack(data[:,1])
# 做下小统计
#for i in range(5):
#    #subject_num = data[i,2].reshape(-1)
#    #print(len(set(subject_num)))
#    ShR_num = len(np.argwhere(data[i,3]==20))
#    print(ShR_num)


X = X[:,0:int(signal_len*Fs)]
Y = Y.reshape(-1)
ShR_index = np.argwhere(Y==1).reshape(-1)
NshR_index = np.argwhere(Y==0).reshape(-1)

ShR_ecg = X[ShR_index,:]
NshR_ecg = X[NshR_index,:]

num = 481
plt.plot(NshR_ecg[num,:],linewidth=4,color='black')
plt.show()
cnn_data = np.diff(NshR_ecg)

plt.plot(cnn_data[num,:],linewidth=4,color='black')
plt.show()
# 获取窗口数据
X,timestep = slide_window(X,window,step)
X = np.expand_dims(X,axis=-1)
# 构建模型
input0 = Input(shape=(X.shape[1],X.shape[2],1),name='input0')
conv2d = Conv2D(filters=1,kernel_size=[1,4],strides=1, padding='valid',use_bias=0,name='conv2d',trainable=True)
tensor = conv2d(input0)
tensor = Lambda(lambda x:K.squeeze(x,axis=-1),name='squeeze_0')(tensor)
tensor = Lambda(lambda x:tf.nn.top_k(x,x.shape[2])[0],name='sequential_pooling_drop')(tensor) 
tensor = Lambda(lambda x:K.reverse(x, axes=2),name='sequential_pooling_asend')(tensor) 
tensor = LSTM(units = 40,name='lstm')(tensor) 
tensor = Dense(20,activation ='tanh',name='FC0')(tensor)
prediction = Dense(1, activation='sigmoid',name='output',use_bias='False')(tensor)
model = Model(inputs = input0,outputs = prediction)
# 加载训练好的模型权重 
model.load_weights(r'.\model\CNN_SP_LSTM\\'+str(signal_len)+'s-train_2-fold.h5py')
                   
layer_name ='sequential_pooling_asend'
intermediate_layer_model = Model(input=model.input,output=model.get_layer(layer_name).output)

ShR_X = X[ShR_index,:,:,:]
NshR_X = X[NshR_index,:,:,:]

intermediate_output = intermediate_layer_model.predict(NshR_X)

# 绘制顺序统计量特征
plt.figure(figsize=(4,6))
a = intermediate_output[num,0,:]+1.2
b = intermediate_output[num,1,:]+1.07
d = intermediate_output[num,-1,:]+0.94

plt.plot(a,linewidth = 4.5,color='black')
plt.plot(b,linewidth = 4.5,color='black')
plt.plot(d,linewidth = 4.5,color='black')
plt.scatter([75,75,75],[0.98,1.01,1.04],color='black',s=20)

plt.ylim([0.7,1.34])
plt.axis('off')
plt.show()
# 绘制学习到的状态
layer_name ='lstm'
intermediate_layer_model = Model(input=model.input,output=model.get_layer(layer_name).output)
intermediate_output_NshR = intermediate_layer_model.predict(NshR_X)
intermediate_output_ShR = intermediate_layer_model.predict(ShR_X)

plt.figure(figsize=(4,6))
plt.subplot(211)
plt.plot(np.mean(intermediate_output_NshR[:1500,:],axis=0),color='black',linewidth=4.5)
#plt.title('NshR status')
plt.ylim([-1.1,1])
plt.axis('off')
plt.subplot(212)
plt.plot(np.mean(intermediate_output_ShR[:1500,:],axis=0),color='black',linewidth=4.5)
plt.ylim([-1.1,1])
#plt.title('ShR status')
plt.axis('off')