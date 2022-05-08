import torch
import numpy as np
from PIL import Image
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


def download_MNIST(DownloadRoot, Augment):
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.1307, ], [0.3081, ]),
                                         transforms.Resize(32)
                                         ])

    transform_train = transform_Augment('MNIST') if Augment else transform_test
    MNIST_train = datasets.MNIST(root=DownloadRoot,  # 数据保存路径
                                 train=True,  # 作为训练集
                                 download=True,  # 是否下载该数据集
                                 transform=transform_train
                                 )

    MNIST_test = datasets.MNIST(root=DownloadRoot,  # 数据保存路径
                                train=False,  # 作为测试集
                                download=True,  # 是否下载该数据集
                                transform=transform_test
                                )
    return MNIST_train, MNIST_test


def download_CIFAR10(DownloadRoot, Augment):
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                         ])

    transform_train = transform_Augment('CIFAR10') if Augment else transform_test
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


def MNSIT_Dataloader(batch_size, Augment=True, DownloadRoot=r".\dataSet"):
    MNIST_train, MNSIT_test = download_MNIST(DownloadRoot, Augment)

    if batch_size == 'BGD':
        train_batch_size = MNIST_train.data.shape[0]  # 50000
        test_batch_size = MNSIT_test.data.shape[0]  # 10000
    elif batch_size == 'SGD':
        train_batch_size = 1
        test_batch_size = 1
    else:
        train_batch_size = batch_size
        test_batch_size = batch_size

    train_loader = DataLoader(MNIST_train, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(MNSIT_test, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, 1


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)  # 图片转numpy
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
        mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
        img[mask == 0] = 0  # 椒
        img[mask == 1] = 255  # 盐
        img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
        return img