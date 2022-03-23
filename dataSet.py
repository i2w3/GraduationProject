from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def trans_Compose(resize=None, normalize=None, Random=None):
    trans = [transforms.ToTensor()]
    if Random:
        trans.insert(0, transforms.RandomCrop(32, padding=4))  # 数据增广
        trans.insert(0, transforms.RandomHorizontalFlip())  # 依50%概率水平翻转
        print(f"Dataset enable Random")
    if resize:
        trans.insert(0, transforms.Resize(resize))  # 调整data的size
    if normalize:
        if normalize == "MNIST":
            trans.append(transforms.Normalize([0.1307, ], [0.3081, ]))
        if normalize == "CIFAR10":
            trans.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        print(f"Dataset enable Normalize")
    return transforms.Compose(trans)


# 加载数据集
def load_MNIST(batch_size, resize=None, normalize=None, Random=None):
    if normalize:
        normalize = "MNIST"
    MNIST_train = datasets.MNIST(root=r'.\data',  # 数据保存路径
                                 train=True,  # 作为训练集
                                 download=True,  # 是否下载该数据集
                                 transform=trans_Compose(resize, normalize, Random)
                                 )
    print(f"MNIST训练数据集处理完成")
    MNIST_test = datasets.MNIST(root=r'.\data',  # 数据保存路径
                                train=False,  # 作为测试集
                                download=True,  # 是否下载该数据集
                                transform=trans_Compose(resize, normalize)
                                )
    print(f"MNIST测试数据集处理完成")
    return (DataLoader(MNIST_train, batch_size=batch_size, shuffle=True),
            DataLoader(MNIST_test, batch_size=batch_size, shuffle=False),
            1)


def load_CIFAR10(batch_size, normalize=None, Random=None):
    if normalize:
        normalize = "CIFAR10"
    CIFAR10_train = datasets.CIFAR10(root=r'.\data',  # 数据保存路径
                                     train=True,  # 作为训练集
                                     download=True,  # 是否下载该数据集
                                     transform=trans_Compose(normalize=normalize, Random=Random)
                                     )
    print(f"CIFAR10训练数据集处理完成")
    CIFAR10_test = datasets.CIFAR10(root=r'.\data',  # 数据保存路径
                                    train=False,  # 作为测试集
                                    download=True,  # 是否下载该数据集
                                    transform=trans_Compose(normalize=normalize)
                                    )
    print(f"CIFAR10测试数据集处理完成")
    return (DataLoader(CIFAR10_train, batch_size=batch_size, shuffle=True),
            DataLoader(CIFAR10_test, batch_size=batch_size, shuffle=False),
            3)
