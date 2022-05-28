import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from utils import imshow
from dataSet import MNSIT_Dataloader

num_row = 8
num_column = 8
count = num_row * num_column

trainloader, _, _ = MNSIT_Dataloader(count, Augment=False, DownloadRoot='./')

img, label = iter(trainloader).next()

plt.axis('off')  # 去坐标轴
plt.xticks([])  # 去刻度

imshow(make_grid(img[:count], nrow=num_row, padding=0))
savePath = f'./plot_MNIST {num_row}x{num_column}.png'
plt.savefig(savePath, bbox_inches='tight', pad_inches=0, dpi=1024)
plt.show()
