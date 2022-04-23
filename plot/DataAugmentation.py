import random
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from dataSet import AddGaussianNoise, AddSaltPepperNoise

img_Path = r"./tset.jpg"
img = Image.open(img_Path)
img0 = img.convert("RGB")  # 原图

img1 = transforms.RandomCrop(500, padding=125)(img0)  # 随机裁剪
img2 = transforms.RandomHorizontalFlip(p=1)(img0)  # 水平翻转
img3 = transforms.RandomVerticalFlip(p=1)(img0)  # 垂直翻转
img4 = transforms.RandomRotation(degrees=90)(img0)  # 旋转180度
img5 = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(img0)  # 色彩抖动

a = np.array(img0)
ToTensor = transforms.ToTensor()
GaussianNoise = AddGaussianNoise(0., 0.75)
img6 = ToTensor(a)
img6 = GaussianNoise(img6)  # 高斯噪声
img6 = np.transpose(img6.numpy(), (1, 2, 0))

img7 = AddSaltPepperNoise(0.2)(img0)  # 椒盐噪声

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['SimSun'],
    "font.size": 12,
    "axes.unicode_minus": False,
    "figure.autolayout": True
}
plt.rcParams.update(config)

axs = plt.figure().subplots(2, 4)


def show(ax, i, j, img, title):
    ax[i][j].imshow(img)
    ax[i][j].set_xticks([])
    ax[i][j].set_yticks([])
    ax[i][j].set_xlabel(title)


show(axs, 0, 0, img0, "(a)原始图像")
show(axs, 0, 1, img1, "(b)随机裁剪")
show(axs, 0, 2, img2, "(c)水平翻转")
show(axs, 0, 3, img3, "(d)垂直翻转")

show(axs, 1, 0, img4, "(e)随机旋转")
show(axs, 1, 1, img5, "(f)色彩抖动")
show(axs, 1, 2, img6, "(g)高斯噪声")
show(axs, 1, 3, img7, "(h)椒盐噪声")
plt.savefig('./DataAugmentation.png', bbox_inches='tight', pad_inches=0, dpi=1024)
plt.show()
