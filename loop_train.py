from utils import *
from model.LeNet import Lenet
from model.ResNet import resnet18
from model.SE_ResNet import se_resnet18, twose_resnet18
from dataSet import load_MNIST, load_CIFAR10
from torchsummary import summary

'''
milestones = [80, 120]
gamma=0.1
epochs=160
lr=0.1
'''

# 检查cuda
device = check_Device()

# 设置随机种子，增加PyTorch中模型的可复现性
seed_Setting(0)

# 参数设置
LEARNING_RATE = 0.1
BATCH_SIZE = 64
EPOCHS = 160

# 数据设置
train_loader, valid_loader, channel = load_CIFAR10(BATCH_SIZE, Normalize=True, Random=True, Noise=None)

for net in [resnet18(), se_resnet18()]:
    model = net.to(device)
    summary(model, (channel, 224, 224))
    params = sum(p.numel() for p in list(model.parameters())) / 1e6
    print('#Params: %.1fM' % params)

    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model, optimizer, (_, _), (_, _) = training_loop(model, criterion, optimizer, train_loader, valid_loader, device,
                                                     EPOCHS)
