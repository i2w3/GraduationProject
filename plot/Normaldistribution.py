import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
plt.rcParams['font.size'] = 12  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 期望0，标准差1
mu, sigma = 0, 1

x_axis = np.arange(-10, 10, 0.001)

plt.plot(x_axis, norm.pdf(x_axis, mu, sigma))
plt.xlabel('期望值')
plt.ylabel('概率')
plt.grid(True)
plt.savefig("./标准正态分布.png", bbox_inches='tight', pad_inches=0)
plt.show()
