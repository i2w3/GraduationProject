import torch.nn as nn
from model.SE_ResNet import SEBlock


def Lenet(channel, SE2=None, BN2=None, SE1=None, BN1=None, reduction=16):
    """
    # 常用参数：channel 表示输入的通道，1表示黑白图像，3表示RGB图像；reduction默认16即可
    net = Lenet(channel) # 经典的Lenet-5模型
    net = Lenet(channel, BN=True) # 在第二次卷积后添加一个BN
    net = Lenet(channel, SE=True, BN=True) # 在第二次卷积后添加一个BN和SE Block (作者建议非残差网络先BN后SE)
    net = Lenet(channel, SE=True) # 测试不加BN的SE，结果和经典的Lenet-5模型差别不大
    net = Lenet(channel, FullBN=True) # 两次卷积后都添加BN,用来测试两次conv后都加BN的影响
    """
    net = nn.Sequential()

    # Conv 1
    net.add_module("Conv1", (nn.Conv2d(channel, 6, 5, 1)))
    # in_channels=channel,out_channels=6,kernel_size=5,stride=1, padding=0
    if BN1:
        net.add_module("Bn1", (nn.BatchNorm2d(6, momentum=0.9, eps=1e-5)))
    net.add_module("Activation1", nn.Tanh())
    if SE1:
        net.add_module("SE Block1", SEBlock(6, reduction=reduction))
        print(f"SE1 Reduction={reduction}")


    # Pool 1
    net.add_module("Pool1", nn.AvgPool2d(kernel_size=2, stride=2))

    # Conv 2
    net.add_module("Conv2", (nn.Conv2d(6, 16, 5, 1)))
    if BN2:
        # 非残差网络要用SE Block最好再conv后加个BN
        net.add_module("Bn2", (nn.BatchNorm2d(16, momentum=0.9, eps=1e-5)))
    net.add_module("Activation2", nn.Tanh())
    if SE2:
        net.add_module("SE Block2", SEBlock(16, reduction=reduction))
        print(f"SE2 Reduction={reduction}")

    # Pool 2
    net.add_module("Pool2", nn.AvgPool2d(kernel_size=2, stride=2))

    # FC
    net.add_module("Flatten", nn.Flatten())
    net.add_module("Linear1", nn.Linear(16 * 5 * 5, 120))
    net.add_module("Activation3", nn.Tanh())
    net.add_module("Linear2", nn.Linear(120, 84))
    net.add_module("Activation4", nn.Tanh())
    net.add_module("Linear3", nn.Linear(84, 10))

    return net
