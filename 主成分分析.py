# -*- coding: utf-8 -*-
"""
主成分分析，绘制网络中间结果
"""
import warnings
warnings.filterwarnings("ignore")

import scipy.io as sio
import numpy as np
from my_functions import slide_window
import matplotlib.pyplot as plt
from keras.layers import Dense,Input,Lambda,Conv2D,LSTM
from keras import backend as K
import tensorflow as tf
from keras.models import Model
from sklearn.preprocessing import StandardScaler
std = StandardScaler()


# 模型参数
units = 30
path = 'CNN_SP_LSTM_units='+str(units)
signal_len = 2                     # 信号长度
Fs = 250                           # 信号采样率
window = 0.6                      # 滑动窗宽

k = 0
# 加载数据 
data = sio.loadmat(r'.\ECG_data\public_ECG_5fold_data_with_annotion_1.mat')['data']

cv = np.eye(5) #1是测试集  0是训练集

# 训练集索引
index = np.argwhere(cv[k,]==0).reshape(-1)
X_train = np.vstack(data[index,0])
Y_train = np.vstack(data[index,1]).reshape(-1)
X_train = X_train[:,0:int(signal_len*Fs)]
# 获取窗口数据
X_train = slide_window(X_train,window,step=0.2)[0]

X_train = np.expand_dims(X_train,axis=-1)

def build_model(X_train):
    input0 = Input(shape=(X_train.shape[1],X_train.shape[2],1),name='input0')
    conv2d = Conv2D(filters=1,kernel_size=[1,4],strides=1, padding='valid',use_bias=0,name='conv2d',trainable=True)
    tensor = conv2d(input0)  
    tensor = Lambda(lambda x:K.squeeze(x,axis=-1),name='squeeze_0')(tensor) # 4D 数据压缩为 3D 数据,因为channels=1
    tensor = Lambda(lambda x:tf.nn.top_k(x,x.shape[2])[0],name='sequential_pooling_drop')(tensor) # 在第2个维度上对数据降序排列,即各滑动窗内数据
    tensor = Lambda(lambda x:K.reverse(x, axes=2),name='sequential_pooling_asend')(tensor) # 升序排列
    tensor = LSTM(units = units,name='lstm')(tensor) # 多对一LSTM
    tensor = Dense(20,activation ='tanh',name='FC0')(tensor)
    prediction = Dense(1, activation='sigmoid',name='output',use_bias='False')(tensor)
    return Model(inputs = input0,outputs = prediction)
    
model = build_model(X_train)
#model.load_weights(r'.\model\CNN_SP_LSTM_units\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.h5py')
model.load_weights(r'.\model\CNN_SP_LSTM\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.h5py')

# 中间输出
cnn_window_num = 0
sp_window_num = 0
intermediate_layer_model = Model(input = model.input,output=model.get_layer('squeeze_0').output)
conv_output = intermediate_layer_model.predict(X_train)
conv_output = conv_output[:,cnn_window_num,:]
#conv_output = std.fit_transform(conv_output)

intermediate_layer_model = Model(input = model.input,output=model.get_layer('sequential_pooling_asend').output)
SP_output = intermediate_layer_model.predict(X_train)
SP_output = SP_output[:,sp_window_num,:]
#SP_output = std.fit_transform(SP_output)

intermediate_layer_model = Model(input = model.input,output=model.get_layer('lstm').output)
lstm_output = intermediate_layer_model.predict(X_train)
#lstm_output = std.fit_transform(lstm_output)


intermediate_layer_model = Model(input = model.input,output=model.get_layer('FC0').output)
FC0_output = intermediate_layer_model.predict(X_train)
#￥C0_output = std.fit_transform(FC0_output)

#intermediate_output = np.squeeze(intermediate_output)
def PCA_analysis(data):
    cov_mat = np.cov(data.T)
    #cov_mat = np.cov(lstm_output.T)
    eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)
    #print('\nEigenvalues\n%s' %eigen_vals)
    
    tot = sum(eigen_vals)
    var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
#    cum_var_exp = np.cumsum(var_exp)
#    n = 8 # 主成分个数
    
#    plt.figure(figsize=(8,8*0.618))
#    plt.bar(range(1,n),var_exp[0:n-1],alpha=0.7,align='center',color='blue',
#            edgecolor = 'black',lw=2,label='individual explained variance')
#    #plt.step(range(1,n),cum_var_exp[0:n-1],where='mid',label='cumulative explained variance')
#    plt.plot(range(1,n),cum_var_exp[0:n-1],color='red')
#    plt.scatter(range(1,n),cum_var_exp[0:n-1],s=40,color='red',label='cumulative explained variance')
#    plt.vlines(3,0,cum_var_exp[3],colors = "c", linestyles = "dashed")# 垂直线
#    plt.hlines(cum_var_exp[3],1,3,colors = "c", linestyles = "dashed")# 水平线
#    plt.ylabel('Explained variance ratio',fontsize=14)
#    plt.xlabel('Principal components',fontsize=14)
#    plt.legend(loc='best',fontsize=14)
#    plt.show()
    #特征转换
    eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(reverse=True)
    
    w = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
    
    return w

w_conv = PCA_analysis(conv_output)
w_SP = PCA_analysis(SP_output)
w_lstm = PCA_analysis(lstm_output)
w_FC0 = PCA_analysis(FC0_output)

pca_conv = conv_output.dot(w_conv)
pca_SP = SP_output.dot(w_SP)
pca_lstm = lstm_output.dot(w_lstm)
pca_FC0 = FC0_output.dot(w_FC0)

#pca_data = np.dstack([pca_conv,pca_SP,pca_lstm,pca_FC0])
#import scipy.io as io
#io.savemat('.\important_data\PCA_data', {'pac_data': pca_data,'label':Y_train})
# 画图

#layer_name = ['Output of the convolution layer','Output of the sequential pooling layer',
#              'Output of the LSTM layer','Output of the full connection layer']
#fig = plt.figure(figsize=(c,0.65*c))
##font = {'family':'Times New Roman', 'weight': 'normal', 'size': 16}
#font = { 'weight': 'normal', 'size': 18}
#marker = ['.','v']
#for i,name in enumerate(layer_name):
#    plt.subplot(2,2,i+1)
#    if(i<2):
#        colors = ['b','r']
#        for l,c in zip([0,1],colors):
#            plt.scatter(pac_data[(Y_train==l),0,i],pac_data[Y_train==l,1,i],c=c,label=l,marker=marker[l],s=12)
#            plt.legend(['NshR','ShR'],loc = 'upper right',prop=font)
#    else:
#        colors = ['r','b']
#        for l,c in zip([1,0],colors):
#            plt.scatter(pac_data[(Y_train==l),0,i],pac_data[Y_train==l,1,i],c=c,label=l,marker=marker[l],s=12)   
#            plt.legend(['ShR','NshR'],loc = 'upper right',prop=font)
#    if(i==0 or i ==2):
#        plt.ylabel('PC2',font)
#    if(i==2 or i ==3):
#        plt.xlabel('PC1',font)
#    if(i==0):
#        plt.xlim([-0.5,0.9])
#        plt.ylim([-0.6,0.1])
#        plt.xticks([-0.4,0,0.4,0.8],fontsize = 16)
#        plt.yticks([0.1,-0.1,-0.3,-0.5],fontsize = 16)
#    if(i==1):
#        plt.xlim([-0.1,4])
#        plt.xticks(fontsize = 16)
#        plt.yticks(fontsize = 16)
#    
#    plt.title(name,font)
#fig.tight_layout()
#plt.savefig('PAC.png',dpi=300)
#plt.show()

path = '.\picture\PCA'
c = 6
fig = plt.figure(figsize=(c,0.65*c))
#font = {'family':'Times New Roman', 'weight': 'normal', 'size': 18}
font = { 'weight': 'normal', 'size': 18}

plt.scatter(pca_conv[(Y_train==0),0],pca_conv[Y_train==0,1],c='b',marker='.',s=12)
plt.scatter(pca_conv[(Y_train==1),0],pca_conv[Y_train==1,1],c='r',marker='.',s=12)

plt.legend(['NshR','ShR'],loc = 'upper right',prop=font,labelspacing=0.2,handletextpad=0.001)
plt.xlim([-2.2,3.2])
plt.ylim([-2,2.5])
plt.xticks([-2,-1,0,1,2,3],fontsize = 18)
plt.yticks([-2,-1,0,1,2],fontsize = 18)
plt.xlabel('PC1',font)
plt.ylabel('PC2',font)
fig.tight_layout()

plt.savefig(path+'\cnn_output.png',dpi=600)
plt.show()


fig = plt.figure(figsize=(c,0.65*c))

plt.scatter(pca_SP[(Y_train==0),0],pca_SP[Y_train==0,1],c='b',marker='.',s=12)
plt.scatter(pca_SP[(Y_train==1),0],pca_SP[Y_train==1,1],c='r',marker='.',s=12)

plt.legend(['NshR','ShR'],loc = 'upper right',prop=font,
           labelspacing=0.2,handletextpad=0.001)
plt.xlim([-0.1,4])
plt.ylim([-2,0.8])
plt.xticks([0,1,2,3,4],fontsize = 18)
plt.yticks([-2,-1,0],fontsize = 18)
plt.xlabel('PC1',font)
plt.ylabel('PC2',font)
fig.tight_layout()
plt.savefig(r'.\picture\PCA\sp_output.png',dpi=600,borderpad=0.1)
plt.show()


fig = plt.figure(figsize=(c,0.65*c))
a = plt.scatter(pca_lstm[(Y_train==1),0],pca_lstm[Y_train==1,1],c='r',marker='.',s=12)
b = plt.scatter(pca_lstm[(Y_train==0),0],pca_lstm[Y_train==0,1],c='b',marker='.',s=12)
#plt.legend([b,a],['NshR','ShR'],loc = 'lower left',prop=font)
plt.legend([b,a],['NshR','ShR'],loc = 'lower right',prop=font,labelspacing=0.2,handletextpad=0.001)

plt.xlim([-1.5,2.5])
plt.ylim([-3.2,0.4])
plt.xticks([-1,0,1,2],fontsize = 18)
plt.yticks([-3,-2,-1,0],fontsize = 18)
plt.xlabel('PC1',font)
plt.ylabel('PC2',font)
fig.tight_layout()
plt.savefig(r'.\picture\PCA\lstm_output.png',dpi=600)
plt.show()

fig = plt.figure(figsize=(c,0.65*c))
a = plt.scatter(-1*pca_FC0[(Y_train==1),0],pca_FC0[Y_train==1,1],c='r',marker='.',s=12)
b = plt.scatter(-1*pca_FC0[(Y_train==0),0],pca_FC0[Y_train==0,1],c='b',marker='.',s=12)
plt.legend([b,a],['NshR','ShR'],loc = 'lower right',
           prop=font,labelspacing=0.2,handletextpad=0.001)

plt.xlim([-3.5,4])
plt.ylim([-2.6,0.8])
plt.xticks([-3,-1,1,3],fontsize = 18)
plt.yticks([-2,-1,0],fontsize = 18)
plt.xlabel('PC1',font)
plt.ylabel('PC2',font)
fig.tight_layout()
plt.savefig(r'.\picture\PCA\fc_output.png',dpi=600)
plt.show()
