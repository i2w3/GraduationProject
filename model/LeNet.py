import torch.nn as nn
from utils import check_Device
from torchsummary import summary
from model.SE_ResNet import SEBlock


def LeNet5(channel=3, BN1=None, SE1=None, BN2=None, SE2=None, reduction=16):
    net = nn.Sequential()

    # C1卷积层
    net.add_module("Conv1", nn.Conv2d(channel, 6, 5, 1))  # C1卷积层用的是6@5x5大小的卷积核
    if BN1:
        net.add_module("BN1", (nn.BatchNorm2d(6)))  # 根据判断是否在C1层后面添加批量归一化层
    net.add_module("Activation1", nn.Tanh())        # 激活函数是Tanh()
    if SE1:
        net.add_module("SE Block1", SEBlock(6))     # 根据判断是否添加SE Block

    # S2池化层
    net.add_module("Pool2", nn.AvgPool2d(kernel_size=2, stride=2))  # S2池化层用的是2x2大小，步长为2的平均池化窗口

    # C3卷积层
    net.add_module("Conv3", nn.Conv2d(6, 16, 5, 1))  # C3卷积层用的是16@5x5大小的卷积核
    if BN2:
        net.add_module("BN2", (nn.BatchNorm2d(16)))
    net.add_module("Activation2", nn.Tanh())
    if SE2:
        net.add_module("SE Block2", SEBlock(16))

    # S4池化层
    net.add_module("Pool4", nn.AvgPool2d(kernel_size=2, stride=2))  # S4池化层

    # C5、FC、Output
    net.add_module("Flatten", nn.Flatten())
    net.add_module("Linear1", nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                            nn.Tanh()))
    net.add_module("Linear2", nn.Sequential(nn.Linear(120, 84),
                                            nn.Tanh()))
    net.add_module("Linear3", nn.Linear(84, 10))

    return net


def modern_LeNet5(channel=3, BN1=None, SE1=None, BN2=None, SE2=None, reduction=16, stride=2):
    if stride == 1:
        Linear1In = 16 * 22 * 22
    else:
        Linear1In = 16 * 5 * 5


    net = nn.Sequential()

    # C1卷积层
    net.add_module("Conv1", nn.Sequential(nn.Conv2d(channel, 6, 3, 1),
                                          nn.Conv2d(6, 6, 3, 1)))  # C1卷积层用的是两个6@3x3大小的卷积核
    if BN1:
        net.add_module("BN1", (nn.BatchNorm2d(6)))  # 根据判断是否在C1层后面添加批量归一化层
    net.add_module("Activation1", nn.ReLU())          # 激活函数为Relu()
    if SE1:
        net.add_module("SE Block1", SEBlock(6))     # 根据判断是否添加SE Block

    # S2池化层
    net.add_module("Pool2", nn.MaxPool2d(kernel_size=2, stride=stride))  # S2池化层用的是2x2大小，步长为stride的最大池化窗口

    # C3卷积层
    net.add_module("Conv3", nn.Sequential(nn.Conv2d(6, 16, 3, 1),
                                          nn.Conv2d(16, 16, 3, 1)))  # C3卷积层用的是两个16@3x3大小的卷积核
    if BN2:
        net.add_module("BN2", (nn.BatchNorm2d(16)))
    net.add_module("Activation2", nn.ReLU())
    if SE2:
        net.add_module("SE Block2", SEBlock(16))

    # S4池化层
    net.add_module("Pool4", nn.MaxPool2d(kernel_size=2, stride=stride))  # S4池化层

    # C5、FC、Output
    net.add_module("Flatten", nn.Flatten())
    net.add_module("Linear1", nn.Sequential(nn.Linear(Linear1In, 120),
                                            nn.ReLU()))
    net.add_module("Linear2", nn.Sequential(nn.Linear(120, 84),
                                            nn.ReLU()))
    net.add_module("Linear3", nn.Linear(84, 10))

    return net


def TestNet(net, channel=3):
    model = net.to(check_Device())
    summary(model, (channel, 32, 32))

# TestNet(LeNet5(3))
# TestNet(modern_LeNet5(3))
# TestNet(modern_LeNet5(3, stride=1))