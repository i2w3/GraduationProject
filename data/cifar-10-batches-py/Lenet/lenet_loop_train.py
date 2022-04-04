import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from model.LeNet import Lenet, modern_Lenet
from dataSet import load_MNIST, load_CIFAR10
from utils import check_Device, seed_Setting, training_loop

# 检查cuda
device = check_Device()

# 设置随机种子，增加PyTorch中模型的可复现性
seed_Setting(0)

# 参数设置
LEARNING_RATE = 0.1
BATCH_SIZE = 128
EPOCHS = 100

# 数据设置
train_loader, valid_loader, channel = load_CIFAR10(BATCH_SIZE, Normalize=True, Random=True, Noise=None)

classic_Model = [Lenet(),  # 使用AvgPool池化和Tanh激活的经典LeNet5
                 Lenet(SE2=True),  # 在第二个卷积层后添加SE Block
                 Lenet(BN2=True),  # 在第二个卷积层后添加BatchNorm
                 Lenet(BN2=True, SE2=True),  # 在第二个卷积层后添加BatchNorm、SE Block
                 Lenet(BN1=True, SE1=True)]  # 在第一个卷积层后添加BatchNorm、SE Block

modern_Model = [modern_Lenet(),  # 使用MaxPool池化和Relu激活的LeNet5
                modern_Lenet(BN2=True, SE2=True),
                modern_Lenet(BN1=True, SE1=True)]

for net in classic_Model + modern_Model:
    model = net.to(device)
    summary(model, (channel, 32, 32))
    params = sum(p.numel() for p in list(model.parameters())) / 1e6
    print('#Params: %.1fM' % params)

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model, optimizer = training_loop(model, criterion, optimizer, train_loader, valid_loader, device, EPOCHS)
