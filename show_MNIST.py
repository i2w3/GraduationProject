import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from utils import imshow
from dataSet import load_MNIST


num_row = 16
num_column = 8
count = num_row * num_column

trainloader, _, _ = load_MNIST(count, Resize=32)

img, label = iter(trainloader).next()

plt.axis('off')  # 去坐标轴
plt.xticks([])  # 去刻度

imshow(make_grid(img[:count], nrow=num_row, padding=0))
plt.savefig('./data/MNIST.png', bbox_inches='tight', pad_inches=0, dpi=1024)
plt.show()
