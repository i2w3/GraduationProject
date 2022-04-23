# 导入相关库
import matplotlib.pyplot as plt
import numpy as np

# 函数


start = -10  # 输入需要绘制的起始值（从左到右）
stop = 10  # 输入需要绘制的终点值
step = 0.02  # 输入步长
fig = plt.figure(figsize=(10, 5))

ax1 = plt.subplot(1, 2, 1)
num = (stop - start) / step  # 计算点的个数
g = lambda z: 1 / (1 + np.exp(-x))
x = np.linspace(start, stop, int(num))
y = g(x)
l1 = ax1.plot(x, y, label='Sigmoid')

g = lambda z: (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
x = np.linspace(start, stop, int(num))
y = g(x)
l2 = ax1.plot(x, y, label='Tanh')

lns = l1 + l2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left')

plt.grid(True)  # 显示网格

ax2 = plt.subplot(1, 2, 2)

g = lambda z: np.maximum(0, z)
x = np.linspace(start, stop, int(num))
y = g(x)
ax2.plot(x, y, label='Relu')

plt.grid(True)  # 显示网格

plt.legend()  # 显示旁注#注意：不会显示后来再定义的旁注
plt.savefig("./" + "Activation.png", bbox_inches='tight', pad_inches=0)

plt.show()
