import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from model.ResNet import resnet18, se_resnet18
from dataSet import CIFAR10_Dataloader
from utils import check_Device, real_Time, get_LearningRate, get_Accuracy, time_Stamp, top_Err, data_Save

# 检查cuda
device = check_Device()

# 设置随机种子，增加PyTorch中模型的可复现性
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 参数设置
LEARNING_RATE = 0.1
BATCH_SIZE = 128
EPOCHS = 100
PRINT_EVERY = 1
MILESTONES = list(map(int, [EPOCHS * 0.5, EPOCHS * 0.75]))

# 数据设置
train_loader, valid_loader, channel = CIFAR10_Dataloader(batch_size=BATCH_SIZE, Augment=True)

for net in [se_resnet18(), resnet18()]:
    model = net.to(device)
    summary(model, (channel, 224, 224))
    params = sum(p.numel() for p in list(model.parameters())) / 1e6
    print('#Params: %.1fM' % params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)

    unix_timestamp = str(time_Stamp())

    print(f'{real_Time()} --- '
          f'Start training loop\t'
          f'training on: {device}')  # 打印训练设备

    # 保存数据
    train_acces = []
    valid_acces = []
    train_losses = []
    valid_losses = []

    # 开始循环训练模型
    for epoch in range(0, EPOCHS):
        # 训练一次
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

        train_loss = running_loss / len(train_loader.dataset)

        train_losses.append(train_loss)

        # 验证一次
        with torch.no_grad():
            model.eval()
            running_loss = 0

            for X, y_true in valid_loader:
                X = X.to(device)
                y_true = y_true.to(device)

                # Forward pass and record loss
                y_hat = model(X)
                loss = criterion(y_hat, y_true)
                running_loss += loss.item() * X.size(0)

            valid_loss = running_loss / len(valid_loader.dataset)
            valid_losses.append(valid_loss)

        if epoch % PRINT_EVERY == (PRINT_EVERY - 1):
            train_acc = get_Accuracy(model, train_loader, device=device)
            valid_acc = get_Accuracy(model, valid_loader, device=device)

            # 输出本次训练的数据
            print(f'{real_Time()} --- '
                  f'Epoch: {epoch + 1}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')
            train_acces.append(train_acc)
            valid_acces.append(valid_acc)

            old_lr = get_LearningRate(optimizer)
            scheduler.step()
            new_lr = get_LearningRate(optimizer)
            if old_lr != new_lr:
                print(f'\t\t\t\t\t\t'
                      f'Learning Rate from {old_lr:.3f} to {new_lr:.3f}')

    # 循环训练结束，计算top1err和top5err，根据时间戳保存数据并绘图

    top1err, top5err = top_Err(model, valid_loader, device)
    print(f'{real_Time()} --- '
          f'Time Stamp: {unix_timestamp}\t'
          f'top1 error: {top1err:.4f}\t'
          f'top5 error: {top5err:.4f}')
    data_Save(model, train_losses, valid_losses, train_acces, valid_acces, unix_timestamp)
'''
milestones = [80, 120]
gamma=0.1
epochs=160
lr=0.1
'''
