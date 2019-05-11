import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt

# Hyper Parameters
EPOCH = 1 
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root = 'data/mnist',
    train = True,
    transform=torchvision.transforms.ToTensor(), 
    # (0, 1) (0-255)将下载的数据改成我需要的Tensor 放在transform 
    download=DOWNLOAD_MNIST
    )
    # plot one example
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()



class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # (1, 28, 28)
                in_channels = 1,
                out_channels = 16, # filter个数 进行扫描操作
                kernel_size = 5, # filter  5 * 5 像素点
                stride=1, # m每隔多少步跳一下
                padding=2, # 把图片周围围上0  if stride = 1 , padding = (kernel_size - 1)/2

            ),#  ->(16, 28, 28)
            nn.ReLU(),#  ->(16, 28, 28)
            nn.MaxPool2d(kernel_size = 2),# 筛选重要的信息#  ->(16, 14, 14)
        )
        self.conv2 = nn.Sequential(#  ->(16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),#  ->(32, 14, 14)
            nn.ReLU(),#->(32, 14, 14)
            nn.MaxPool2d(2)#->(32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x  = self.conv1(x)
        x = self.conv2(x)   # （batch, 32, 7, 7）
        x = x.view(x.size(0),-1)  # （batch, 32* 7* 7）
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)