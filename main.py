from utils import *
from model.LeNet import Lenet
from model.ResNet import resnet18
from model.SE_ResNet import se_resnet18
from dataSet import load_MNIST, load_CIFAR10
from torchsummary import summary

# 检查cuda
device = check_Device()

# 设置随机种子，增加PyTorch中模型的可复现性
seed_Setting(0)

# 参数设置
LEARNING_RATE = 0.1
BATCH_SIZE = 128
EPOCHS = 130

# 数据设置
# train_loader, valid_loader, channel = load_MNIST(BATCH_SIZE, resize=(32, 32))
train_loader, valid_loader, channel = load_CIFAR10(BATCH_SIZE, normalize=True, Random=True)

net = resnet18(device)
# net = se_resnet18(device)
# net = Lenet(channel)
# net = Lenet(channel, BN2=True)
# net = Lenet(channel, SE2=True, BN2=True, reduction=16)
# net = Lenet(channel, SE=True)
# net = Lenet(channel, SE1=True, BN1=True, SE2=True, BN2=True)
# net = Lenet(channel, BN1=True, BN2=True)


# summary(model, (channel, 32, 32))
summary(net, (channel, 224, 224))
params = sum(p.numel() for p in list(net.parameters())) / 1e6
print('#Params: %.1fM' % params)

# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
net, optimizer, (_, _), (_, _) = training_loop(net, criterion, optimizer, train_loader, valid_loader, device,EPOCHS)
