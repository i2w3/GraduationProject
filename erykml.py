import torch

from torchsummary import summary
import time
from d2l import torch as d2l
from datetime import datetime

from vision import *
from dataSet import *
from model.LeNet import *

# 检查cuda和cudnn加速
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# 增加PyTorch中模型的可复现性
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 参数设置
LEARNING_RATE = 0.001
BATCH_SIZE = 100
N_EPOCHS = 50


def get_accuracy(net, data_iter, device=None):
    # 使用GPU计算模型在数据集上的精度
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    # 训练损失总和，词元数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(train_loader, model, criterion, optimizer, device):
    """
    Function for the training step of the training loop
    """

    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    """
    Function for the validation step of the training loop
    """

    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    """
    Function defining the entire training loop
    """

    print(f'{datetime.now().time().replace(microsecond=0)} --- '
          f'Start training loop\t'
          f'training on: {device}')

    # set objects for storing metrics
    train_losses = []
    valid_losses = []
    train_acces = []
    valid_acces = []

    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch + 1}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')
            train_acces.append(train_acc)
            valid_acces.append(valid_acc)
    unix_timestamp = str(int(time.time()))
    print(f'{datetime.now().time().replace(microsecond=0)} --- '
          f'当前时间戳为: {unix_timestamp}')
    # plot_es(epochs, train_losses, valid_losses, "loss", unix_timestamp)
    # plot_es(epochs, train_acces, valid_acces, "acc", unix_timestamp)
    full_plot(epochs, train_losses, valid_losses, train_acces, valid_acces, unix_timestamp)
    saveNpy(unix_timestamp, train_losses, valid_losses, train_acces, valid_acces)
    return model, optimizer, (train_losses, valid_losses), (train_acces, valid_acces)


# define the data loaders
# train_loader, valid_loader, channel = load_MNIST(BATCH_SIZE, resize=(32, 32))
# train_loader, valid_loader, channel = load_CIFAR10(BATCH_SIZE, resize=(32, 32),normalize=(0.5, 0.5, 0.5))
train_loader, valid_loader, channel = load_CIFAR10(BATCH_SIZE, resize=(32, 32), Random=True)

# net = Lenet(channel)
# net = Lenet(channel, BN2=True)
net = Lenet(channel, SE2=True, BN2=True, reduction=16)
# net = Lenet(channel, SE=True)

model = net.to(DEVICE)

summary(model, (channel, 32, 32))

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
model, optimizer, (train_loss, valid_loss), (train_acc, valid_acc) = training_loop(model, criterion, optimizer,
                                                                                   train_loader, valid_loader,
                                                                                   N_EPOCHS, DEVICE)
