import torch
from dataSet import load_MNIST, load_CIFAR10
from utils import *

# 检查cuda
device = check_Device()
# 加载
net = torch.load('./png/1648151594/model.pt')
net.eval()

# top1 acc 和 top5 acc
correct_1 = 0.0
correct_5 = 0.0
total = 0

_, valid_loader, _ = load_CIFAR10(128, normalize=True, Random=True)

with torch.no_grad():
    for n_iter, (image, label) in enumerate(valid_loader):

        image = image.to(device)
        label = label.to(device)

        output = net(image)
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        # compute top 5
        correct_5 += correct[:, :5].sum()

        # compute top1
        correct_1 += correct[:, :1].sum()


print("Top 1 err: ", 1 - correct_1 / len(valid_loader.dataset))
print("Top 5 err: ", 1 - correct_5 / len(valid_loader.dataset))
print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
