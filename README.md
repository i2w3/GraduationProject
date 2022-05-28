# GraduationProject

## 程序目录
```tree
./
│  README.md
│  dataSet.py                       # 用来下载MNSIT、CIFAR-10数据集，并生成dataloader
│  lenet_loop_train_MNIST.py        # LeNet5 分类 MNIST
│  lenet_loop_train_CIFAR10.py      # LeNet5 分类 CIFAR-10
│  resnet_loop_train_MNIST.py       # ResNet18 分类 MNIST
│  resnet_loop_train_CIFAR10.py     # ResNet18 分类 CIFAR-10
│  plot data.py                     # 重绘指定目录下的所有训练过程图
│  utils.py                         # 存放一些多次调用的函数
│
├─data                              # 论文所有的数据
│
├─dataSet                           # 数据集的保存路径
│
├─model                             # 存放LeNet5、ResNet18及SE Block的构建代码
│  │  LeNet.py
│  │  ResNet.py
│  └─ SEAttention.py
│
└─plot                              # 一些绘图函数
```

## 简要说明
《基于卷积神经网络的图像识别研究》，主要包含数据增强、网络结构改进、优化训练算法

## 创新点
将挤压激励模块SE Block引入LeNet5中，并通过实验得出最佳的嵌入位置

## 训练代码说明
主要的训练代码在xxx_loop_train_yyy.py中，基本流程如下
```python
# 超参数设置
LEARNING_RATE = 0.1
BATCH_SIZE = 128
EPOCHS = 100
PRINT_EVERY = 1
MILESTONES = list(map(int, [EPOCHS * 0.5, EPOCHS * 0.75]))

# 读取数据集
train_loader, valid_loader, channel = yyy_Dataloader(batch_size=BATCH_SIZE, Augment=True)

# 构建网络模型
nets = [xxx]

# 损失函数和优化训练算法
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
#可选：学习率调整策略
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)

# 开始循环训练模型
for epoch in range(0, EPOCHS):
    # 训练一次
    model.train()

    for X, y_true in train_loader:
        optimizer.zero_grad() # 优化器梯度清零

        X = X.to(device)
        y_true = y_true.to(device)

        # 预测结果，计算训练损失
        y_hat = model(X)
        loss = criterion(y_hat, y_true)

        # 反向传播
        loss.backward()     # 计算梯度
        optimizer.step()    # 更新参数

    # 验证一次
    with torch.no_grad():
        model.eval()

        for X, y_true in valid_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            # 预测结果，计算验证损失
            y_hat = model(X)
            loss = criterion(y_hat, y_true)

    # 可选：学习率更新
    scheduler.step()
```
