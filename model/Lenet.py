import torch.nn as nn
from model.SE_Layer import SELayer


def Lenet(channel, SE=None):
    Lenet = nn.Sequential(nn.Conv2d(channel, 6, 5, 1), nn.ReLU(),
                          ## in_channels=channe,out_channels=6,kernel_size=5,stride=1, padding=0
                          nn.AvgPool2d(kernel_size=2, stride=2),
                          nn.Conv2d(6, 16, 5, 1), nn.ReLU(),
                          nn.AvgPool2d(kernel_size=2, stride=2))
    if SE:
        Lenet.add_module("SE Layer", SELayer(16))

    Lenet.add_module("Flatten", nn.Flatten())
    Lenet.add_module("Linear1", nn.Linear(16 * 5 * 5, 120))
    Lenet.add_module("ReLU", nn.ReLU())
    Lenet.add_module("Linear2", nn.Linear(120, 84))
    Lenet.add_module("ReLU", nn.ReLU())
    Lenet.add_module("Linear3", nn.Linear(84, 10))
    return Lenet


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
