import numpy as np
import matplotlib.pyplot as plt

# 定义 Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成一系列 x 值，例如从 -5 到 5 的100个均匀间隔的点
x = np.linspace(-5, 5, 100)

# 计算对应的 Sigmoid 函数值
y = sigmoid(x)

# 绘制 Sigmoid 函数曲线
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Sigmoid Function', color='b', linewidth=2)

# 添加标题和标签
plt.title('Sigmoid Function', fontsize=15)
plt.xlabel('x', fontsize=12)
plt.ylabel('sigmoid(x)', fontsize=12)

# 显示网格线
plt.grid(True, linestyle='--', alpha=0.5)

# 显示图例
plt.legend()

# 显示图形
plt.show()
