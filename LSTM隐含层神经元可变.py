# -*- coding: utf-8 -*-
"""
可除颤心律信号识别
卷积+顺序池化+LSTM网络(隐含层神经元个数可变)
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.chdir('D:\project\python_code\ECG_project_2019')
import time;
time_start = time.time();
import pickle
import numpy as np
import scipy.io as sio  

from keras.layers import Dense,Input,Lambda,Conv2D,LSTM
from keras.models import Model
from keras import backend as K
import tensorflow as tf

from sklearn.metrics import roc_curve

from my_loss import roc_auc_score
from my_functions import calc_ber,calc_auc,auc,slide_window
# 重要模型参数
signal_len = 8
window  = 0.6      # 滑动窗宽(s)
step = 0.2         # 滑动窗移动步长      
Fs = 250           # 信号采样率
units_list = [20,30,40,50,60]     # LSTM内部全连接层神经元个数
# 加载数据
data = sio.loadmat(r'.\ECG_data\public_ECG_5fold_data_with_annotion_1.mat')['data']
cv = np.eye(5) #1是测试集  0是训练集
for units in units_list:
   # print(units)
    path = 'CNN_SP_LSTM\\CNN_SP_LSTM_units='+str(units)
    for i in  ['model','important_data']:
        if not os.path.exists(i+'\\'+path):
            os.makedirs(i+'\\'+path)
    
    for k in range(len(cv)):   #交叉验证
        # 训练集索引
        index = np.argwhere(cv[k,]==0).reshape(-1)
        X_train = np.vstack(data[index,0]) #第 0 列是原始信号
        Y_train = np.vstack(data[index,1]).reshape(-1) # 第 1 列是标记信息
        # 测试集索引
        index = np.argwhere(cv[k,]==1).reshape(-1)
        X_test = np.vstack(data[index,0])
        Y_test = np.vstack(data[index,1]).reshape(-1)
        # 打乱测试集
        permutation = np.random.permutation(X_test.shape[0])
        X_test = X_test[permutation,:]
        Y_test = Y_test[permutation]
        print('%d fold 训练集 ShR:%d,NshR:%d'%((k+1),len(np.argwhere(Y_train==1)),len(np.argwhere(Y_train==0))))
        print('%d fold 测试集 ShR:%d,NshR:%d'%((k+1),len(np.argwhere(Y_test==1)),len(np.argwhere(Y_test==0))))
        # 取前signal_len的信号
        X_train = X_train[:,0:int(signal_len*Fs)]
        X_test = X_test[:,0:int(signal_len*Fs)]
        # 获取滑动窗数据,输入 2D 数据,维度(None,signal_len*Fs),输出为 3D 数据(None,滑动窗个数,滑动窗内采样点个数)
        X_train = slide_window(X_train,window,step)[0]
        X_test = slide_window(X_test,window,step)[0]
        # 增加一维度以适应二维卷积的输入格式
        X_train = np.expand_dims(X_train,axis=-1)
        X_test = np.expand_dims(X_test,axis=-1)
        # 构建卷积-顺序池化长短时记忆神经网络
        #‘channels_last’模式下，输入形如（samples，rows，cols，channels）的4D张量
        input0 = Input(shape=(X_train.shape[1],X_train.shape[2],1),name='input0')
        conv2d = Conv2D(filters=1,kernel_size=[1,4],strides=1, padding='valid',use_bias=0,name='conv2d',trainable=True)
        tensor = conv2d(input0)
        # 设定卷积网络的初值
        init_cnn_weight = []
        init_cnn_kernel = np.array([[   [[-1]],[[1]],[[0]],[[0]]  ]]) #一阶差分初始权重
        init_cnn_weight.append(init_cnn_kernel)
        conv2d.set_weights(init_cnn_weight)
      
        tensor = Lambda(lambda x:K.squeeze(x,axis=-1),name='squeeze_0')(tensor) # 4D 数据压缩为 3D 数据,因为channels=1
        tensor = Lambda(lambda x:tf.nn.top_k(x,x.shape[2])[0],name='sequential_pooling_drop')(tensor) # 在第2个维度上对数据降序排列,即各滑动窗内数据
        tensor = Lambda(lambda x:K.reverse(x, axes=2),name='sequential_pooling_asend')(tensor) # 升序排列
        tensor = LSTM(units = units,name='lstm')(tensor) # 多对一LSTM
        tensor = Dense(20,activation ='tanh',name='FC0')(tensor)
        prediction = Dense(1, activation='sigmoid',name='output',use_bias='False')(tensor)
        model = Model(inputs = input0,outputs = prediction)
        # 中间输出
    #    layer_name = 'sequential_pooling_asend'
    #    intermediate_layer_model = Model(input=model.input,
    #                                 output=model.get_layer(layer_name).output)
    #    intermediate_output = intermediate_layer_model.predict(X_train)
    #    intermediate_output = np.squeeze(intermediate_output)
        # 训练模型
        model.compile(loss = roc_auc_score ,optimizer ='adam',metrics=[auc])
        history = model.fit(X_train,Y_train,batch_size=256,epochs=100,validation_data=(X_test,Y_test))
        # 保存模型
        model.save(r'.\model\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.h5py')
        #cnn_weight = np.squeeze(conv2d.get_weights())
        # 预测
        prob_trainY = model.predict(X_train)
        prob_testY = model.predict(X_test)  
        # ROC曲线 tpr = Se,fpr = 1-Sp Youden_index = Se+Sp-1 = tpr-fpr(约登指数)
        fpr,tpr,thresholds = roc_curve(Y_train,prob_trainY,1,drop_intermediate=False)
        Youden_index = tpr-fpr
        # 概率转换成预测标签
        thr = thresholds[np.where(Youden_index == np.max(Youden_index))]
        pred_trainY = (prob_trainY >= thr).astype('int8')
        pred_testY = (prob_testY >= thr).astype('int8')
        # 评价指标
        Se_tr,Sp_tr,Ber_tr = calc_ber(Y_train,pred_trainY)  
        Se_te,Sp_te,Ber_te = calc_ber(Y_test,pred_testY)
        Auc_tr = calc_auc(Y_train,prob_trainY,1)
        Auc_te = calc_auc(Y_test,prob_testY,1)        
        # 绘制代价函数曲线
        train_loss = history.history['loss']
        test_loss = history.history['val_loss']
          
        train_auc = history.history['auc']
        test_auc = history.history['val_auc']
        # 保存重要数据
        important_data = {'train_loss':train_loss,'test_loss':test_loss,
                          'Se':Se_te,'Sp':Sp_te,'Auc':Auc_te,'Ber':Ber_te,
                          'Se_tr':Se_tr,'Sp_tr':Sp_tr,'Auc_tr':Auc_tr,'Ber_tr':Ber_tr,
                          'prob_trainY':prob_trainY,
                          'prob_testY': prob_testY,
                          'threshold':thr,'train_auc':train_auc,'test_auc':test_auc}
        data_path = r'.\important_data\\'+path+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
        with open(data_path, 'wb') as fw:
            pickle.dump(important_data, fw)    
    
#显示结果
for units in units_list:
    Se = []; Sp = []; Ber = []; Auc = []
    for k in range(5):
       
        data_path = '.\important_data\CNN_SP_LSTM\CNN_SP_LSTM_units='+str(units)+'\\'+str(signal_len)+'s-train_'+str(k+1)+'-fold.pickle'
        with open(data_path, 'rb') as f:
            a = pickle.load(f)
            Se.append(a['Se']);Sp.append(a['Sp']);Ber.append(a['Ber']);Auc.append(a['Auc'])  
    print('units = %d'%units)
    print('%ds ecg mean_Se:%.2f%%'%(signal_len,np.mean(Se)))
    print('%ds ecg mean_Sp:%.2f%%'%(signal_len,np.mean(Sp)))
    print('%ds ecg mean_Ber:%.2f%%'%(signal_len,np.mean(Ber)))
    print('%ds ecg mean_Auc:%.3f'%(signal_len,np.mean(Auc)))
    print('\n',end="")
time = time.time()-time_start
print('程序运行 %d 小时 %d 分'%(time//3600,time%60))   