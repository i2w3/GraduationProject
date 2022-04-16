import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from utils import imshow
from dataSet import load_CIFAR10


num_row = 16
num_column = 8
count = num_row * num_column

trainloader, _, _ = load_CIFAR10(count)

img, label = iter(trainloader).next()

plt.axis('off')  # 去坐标轴
plt.xticks([])  # 去刻度

imshow(make_grid(img[:count], nrow=num_row, padding=1))
plt.savefig('./data/CIFAR10.png', bbox_inches='tight', pad_inches=0, dpi=1024)
plt.show()
