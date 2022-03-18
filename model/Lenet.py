import torch.nn as nn
from model.SE_Layer import SELayer


def Lenet(channel, SE=None, BN=None, reduction=32):
    net = nn.Sequential()

    # Conv 1
    net.add_module("Conv1", (nn.Conv2d(channel, 6, 5, 1)))
    # in_channels=channel,out_channels=6,kernel_size=5,stride=1, padding=0
    net.add_module("Activation1", nn.Tanh())

    # Pool 1
    net.add_module("Pool1", nn.AvgPool2d(kernel_size=2, stride=2))

    # Conv 2
    net.add_module("Conv2", (nn.Conv2d(6, 16, 5, 1)))
    if BN:
        # 非残差网络要用SE Block最好再conv后加个BN
        net.add_module("Bn2", (nn.BatchNorm2d(16, momentum=0.9, eps=1e-5)))
    net.add_module("Activation2", nn.Tanh())
    if SE:
        net.add_module("SE Layer", SELayer(16, reduction=reduction))

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
