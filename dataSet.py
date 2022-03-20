from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def trans_Compose(resize=None, normalize=None, Random=None):
    trans = [transforms.ToTensor()]  # 转换data的数据类型为Torch Tensor
    if resize:
        trans.insert(0, transforms.Resize(resize))  # 调整data的size
    if Random:
        trans.insert(0, transforms.RandomHorizontalFlip())  # 数据增广
        trans.insert(0, transforms.RandomCrop(32, padding=4))  # 数据增广
    if normalize:
        trans.append(transforms.Normalize(normalize))  # 需不需要标准化
    return transforms.Compose(trans)


# 加载数据集
def load_MNIST(batch_size, resize=None, normalize=None, Random=None):
    if normalize:
        normalize = ((0.1307,), (0.3081,))
    MNIST_train = datasets.MNIST(root=r'.\data',  # 数据保存路径
                                 train=True,  # 作为训练集
                                 download=True,  # 是否下载该数据集
                                 transform=trans_Compose(resize, normalize, Random)
                                 )
    MNIST_test = datasets.MNIST(root=r'.\data',  # 数据保存路径
                                train=False,  # 作为测试集
                                download=True,  # 是否下载该数据集
                                transform=trans_Compose(resize, normalize)
                                )
    return (DataLoader(MNIST_train, batch_size=batch_size, shuffle=True),
            DataLoader(MNIST_test, batch_size=batch_size, shuffle=False),
            1)


def load_CIFAR10(batch_size, resize=None, normalize=None, Random=None):
    if normalize:
        normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    CIFAR10_train = datasets.CIFAR10(root=r'.\data',  # 数据保存路径
                                     train=True,  # 作为训练集
                                     download=True,  # 是否下载该数据集
                                     transform=trans_Compose(resize, normalize, Random)
                                     )
    CIFAR10_test = datasets.CIFAR10(root=r'.\data',  # 数据保存路径
                                    train=False,  # 作为测试集
                                    download=True,  # 是否下载该数据集
                                    transform=trans_Compose(resize, normalize)
                                    )
    return (DataLoader(CIFAR10_train, batch_size=batch_size, shuffle=True),
            DataLoader(CIFAR10_test, batch_size=batch_size, shuffle=False),
            3)
