from utils import *
from model.LeNet import *
from dataSet import load_MNIST, load_CIFAR10

from torchsummary import summary

# 检查cuda
DEVICE = check_Device()

# 设置随机种子，增加PyTorch中模型的可复现性
seed_Setting(0)

# 参数设置
LEARNING_RATE = 0.001
BATCH_SIZE = 100
N_EPOCHS = 50

# 数据设置
# train_loader, valid_loader, channel = load_MNIST(BATCH_SIZE, resize=(32, 32))
train_loader, valid_loader, channel = load_CIFAR10(BATCH_SIZE, resize=(32, 32), normalize=True, Random=True)


net = Lenet(channel)
# net = Lenet(channel, BN2=True)
# net = Lenet(channel, SE2=True, BN2=True, reduction=16)
# net = Lenet(channel, SE=True)
# net = Lenet(channel, SE1=True, BN1=True, SE2=True, BN2=True)
# net = Lenet(channel, BN1=True, BN2=True)
model = net.to(DEVICE)

summary(model, (channel, 32, 32))
params = sum(p.numel() for p in list(net.parameters())) / 1e6
print('#Params: %.1fM' % params)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
model, optimizer, (train_loss, valid_loss), (train_acc, valid_acc) = training_loop(model, criterion, optimizer,
                                                                                   train_loader, valid_loader,
                                                                                   N_EPOCHS, DEVICE)
