"""
pytorch 学习
可除颤心律识别
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import os
import scipy.io as sio  
import numpy as np

from torchsummary import summary
# torch.manual_seed(1)

EPOCH = 5
BATCH_SIZE = 128
LR = 0.001
DOWNLOAD_MNIST = False

signal_len = 2     # 信号长度 (s)
Fs = 250           # 信号采样率
# 加载数据
data = sio.loadmat(r'.\ECG_data\public_ECG_5fold_data_with_annotion_1.mat')['data']
k = 0 
cv = np.eye(5) #1是测试集  0是训练集

index = np.argwhere(cv[k,]==0).reshape(-1)
X_train = np.vstack(data[index,0]) #第 0 列是原始信号
Y_train = np.vstack(data[index,1]).reshape(-1) # 第 1 列是标记信息
# 测试集索引
index = np.argwhere(cv[k,]==1).reshape(-1)
X_test = np.vstack(data[index,0])
Y_test = np.vstack(data[index,1]).reshape(-1)
# 取前signal_len的信号
X_train = X_train[:,0:int(signal_len*Fs)]
X_test = X_test[:,0:int(signal_len*Fs)]

X_train = np.expand_dims(X_train,axis=1)
X_train = torch.from_numpy(X_train)
Y_train = torch.from_numpy(Y_train)

torch_dataset = Data.TensorDataset(X_train, Y_train)

train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=2)

#test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# !!!!!!!! Change in here !!!!!!!!! #
#test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU
#test_y = test_data.test_labels[:2000].cuda()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=2,),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
        self.conv2 = nn.Sequential(nn.Conv2d(5, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),)
        self.out = nn.Linear(32 * 7 * 7, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
    
cnn = CNN().cuda()
summary(cnn,(1,28,28)) #  [batch, height, width, channels]
# !!!!!!!! Change in here !!!!!!!!! #
cnn.cuda()      # Moves all model parameters and buffers to the GPU.

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
losses = []
val_losses = []
for epoch in range(EPOCH):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    for step, (x, y) in enumerate(train_loader):

        # !!!!!!!! Change in here !!!!!!!!! #
        b_x = x.cuda()    # Tensor on GPU
        b_y = y.cuda()    # Tensor on GPU

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
     
        if step % 50 == 0:
           
            test_output = cnn(test_x)

            # !!!!!!!! Change in here !!!!!!!!! #
            pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU
            
            accuracy = torch.sum(pred_y == test_y ).type(torch.FloatTensor) / test_y.size(0)
            #loss = loss_func(test_output, test_y)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
    losses.append(train_loss/len(train_loader))
    # 验证
#    cnn.eval()
#    for step, (x, y) in enumerate(test_x,test_y):
#        b_x = x.cuda()    # Tensor on GPU
#        b_y = y.cuda()    # Tensor on GPU
#    
#        loss = loss_func(cnn(b_x), b_y)
#        val_loss += loss.item()
#    val_losses.append(val_loss/len(test_x))
#    print()
test_output = cnn(test_x[:10])

# !!!!!!!! Change in here !!!!!!!!! #
pred_y = torch.max(test_output, 1)[1].cuda().data # move the computation in GPU

print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
