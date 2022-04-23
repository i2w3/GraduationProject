import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def transform_Augment(Augment):
    if Augment == 'CIFAR10':
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),  # 先四周填充4，在把图像随机裁剪成32*32
                                        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                        # R,G,B每层的归一化用到的均值和方差
                                        ])
    elif Augment == 'MNIST':
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),  # 先四周填充4，在把图像随机裁剪成32*32
                                        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.1307, ], [0.3081, ])
                                        # R,G,B每层的归一化用到的均值和方差
                                        ])
    return transform


def download_CIFAR10(DownloadRoot, Augment):
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                         ])

    if Augment:
        transform_train = transform_Augment('CIFAR10')
    else:
        transform_train = transform_test

    CIFAR10_train = datasets.CIFAR10(root=DownloadRoot,  # 数据保存路径
                                     train=True,  # 作为训练集
                                     download=True,  # 是否下载该数据集
                                     transform=transform_train
                                     )

    CIFAR10_test = datasets.CIFAR10(root=DownloadRoot,  # 数据保存路径
                                    train=False,  # 作为测试集
                                    download=True,  # 是否下载该数据集
                                    transform=transform_test
                                    )
    return CIFAR10_train, CIFAR10_test


def CIFAR10_Dataloader(batch_size, Augment=True, DownloadRoot=r".\dataSet"):
    CIFAR10_train, CIFAR10_test = download_CIFAR10(DownloadRoot, Augment)

    if batch_size == 'BGD':
        train_batch_size = CIFAR10_train.data.shape[0]  # 50000
        test_batch_size = CIFAR10_test.data.shape[0]  # 10000
    elif batch_size == 'SGD':
        train_batch_size = 1
        test_batch_size = 1
    else:
        train_batch_size = batch_size
        test_batch_size = batch_size

    train_loader = DataLoader(CIFAR10_train, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(CIFAR10_test, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, 3
