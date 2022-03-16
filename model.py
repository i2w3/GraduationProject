import torch.nn as nn


def Lenet(channel):
    return nn.Sequential(nn.Conv2d(channel, 6, 5, 1), nn.Sigmoid(),
                         # in_channels=3,out_channels=6,kernel_size=5,stride=1, padding=0
                         nn.AvgPool2d(kernel_size=2, stride=2),
                         nn.Conv2d(6, 16, 5, 1), nn.Sigmoid(),
                         nn.AvgPool2d(kernel_size=2, stride=2),
                         nn.Flatten(),
                         nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                         nn.Linear(120, 84), nn.Sigmoid(),
                         nn.Linear(84, 10)
                         )

def erykml_Lenet(channel):
    return nn.Sequential(nn.Conv2d(channel, 6, 5, 1), nn.Tanh(),
                         # in_channels=3,out_channels=6,kernel_size=5,stride=1, padding=0
                         nn.AvgPool2d(kernel_size=2, stride=2),
                         nn.Conv2d(6, 16, 5, 1), nn.Tanh(),
                         nn.AvgPool2d(kernel_size=2, stride=2),
                         nn.Conv2d(16, 120, 5, 1), nn.Tanh(),
                         nn.Flatten(),
                         nn.Linear(120, 84), nn.Tanh(),
                         nn.Linear(84, 10)
                         )


def modern_Lenet(channel):
    return nn.Sequential(nn.Conv2d(channel, 6, 5, 1), nn.ReLU(),
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.Conv2d(6, 16, 5, 1), nn.ReLU(),
                         nn.MaxPool2d(kernel_size=2, stride=2),
                         nn.Flatten(),
                         nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
                         nn.Linear(120, 84), nn.ReLU(),
                         nn.Linear(84, 10)
                         )


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        # 返回1X1大小的特征图，通道数不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # 全局平均池化，batch和channel和原来一样保持不变
        y = self.avg_pool(x).view(b, c)

        # 全连接层+池化
        y = self.fc(y).view(b, c, 1, 1)

        # 和原特征图相乘
        return x * y.expand_as(x)


def Lenet_SE(channel):
    Lenet_SE = nn.Sequential(nn.Conv2d(channel, 6, 5, 1), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             nn.Conv2d(6, 16, 5, 1), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2))

    Lenet_SE.add_module("SE Layer", SELayer(16))

    Lenet_SE.add_module("Flatten", nn.Flatten())
    Lenet_SE.add_module("Linear1", nn.Linear(16 * 5 * 5, 120))
    Lenet_SE.add_module("ReLU", nn.ReLU())
    Lenet_SE.add_module("Linear2", nn.Linear(120, 84))
    Lenet_SE.add_module("ReLU", nn.ReLU())
    Lenet_SE.add_module("Linear3", nn.Linear(84, 10))
    return Lenet_SE


def SE_Lenet(channel):
    SE_Lenet = nn.Sequential()
    SE_Lenet.add_module("SE Block", SELayer(channel))
    SE_Lenet.add_module("LeNet", Lenet(channel))
    return SE_Lenet

