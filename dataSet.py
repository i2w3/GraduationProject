import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def trans_Compose(Resize=None, Normalize=None, Random=None, Noise=None):
    trans = [transforms.ToTensor()]
    if Random:
        trans.insert(0, transforms.RandomCrop(32, padding=4))  # 数据增广
        trans.insert(0, transforms.RandomHorizontalFlip())  # 依50%概率水平翻转
        print(f"Dataset enable Random")
    if Resize:
        trans.insert(0, transforms.Resize(Resize))  # 调整data的size
    if Normalize:
        if Normalize == "MNIST":
            trans.append(transforms.Normalize([0.1307, ], [0.3081, ]))
        if Normalize == "CIFAR10":
            trans.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]))
        print(f"Dataset enable Normalize")
    if Noise:
        trans.append(transforms.RandomApply([AddGaussianNoise(0., 1.)], p=0.3))
        print(f"Dataset enable Noise")
    return transforms.Compose(trans)


# 加载数据集
def load_MNIST(batch_size, Resize=None, Normalize=None, Random=None, Noise=None):
    global normalize
    if Normalize:
        normalize = "MNIST"
    MNIST_train = datasets.MNIST(root=r'.\data',  # 数据保存路径
                                 train=True,  # 作为训练集
                                 download=True,  # 是否下载该数据集
                                 transform=trans_Compose(Resize, normalize, Random, Noise)
                                 )
    print(f"MNIST训练数据集处理完成")
    MNIST_test = datasets.MNIST(root=r'.\data',  # 数据保存路径
                                train=False,  # 作为测试集
                                download=True,  # 是否下载该数据集
                                transform=trans_Compose(Resize, normalize)
                                )
    print(f"MNIST测试数据集处理完成")
    return (DataLoader(MNIST_train, batch_size=batch_size, shuffle=True),
            DataLoader(MNIST_test, batch_size=batch_size, shuffle=False),
            1)


def load_CIFAR10(batch_size, Resize=None, Normalize=None, Random=None, Noise=None):
    global normalize
    if Normalize:
        normalize = "CIFAR10"
    CIFAR10_train = datasets.CIFAR10(root=r'.\data',  # 数据保存路径
                                     train=True,  # 作为训练集
                                     download=True,  # 是否下载该数据集
                                     transform=trans_Compose(Normalize=normalize, Random=Random, Noise=Noise)
                                     )
    print(f"CIFAR10训练数据集处理完成")
    CIFAR10_test = datasets.CIFAR10(root=r'.\data',  # 数据保存路径
                                    train=False,  # 作为测试集
                                    download=True,  # 是否下载该数据集
                                    transform=trans_Compose(Normalize=normalize)
                                    )
    print(f"CIFAR10测试数据集处理完成")
    return (DataLoader(CIFAR10_train, batch_size=batch_size, shuffle=True),
            DataLoader(CIFAR10_test, batch_size=batch_size, shuffle=False),
            3)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
