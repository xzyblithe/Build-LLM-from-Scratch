"""
第1章代码示例 03：Matplotlib 数据可视化
"""
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# 创建数据
# ================================

x = np.linspace(0, 2*np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

# ================================
# 绘制图形
# ================================

plt.figure(figsize=(15, 10))

# 子图 1: 折线图
plt.subplot(2, 2, 1)
plt.plot(x, y_sin, label='sin(x)', color='blue', linewidth=2)
plt.plot(x, y_cos, label='cos(x)', color='red', linewidth=2, linestyle='--')
plt.title('三角函数')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图 2: 散点图
plt.subplot(2, 2, 2)
x_rand = np.random.randn(100)
y_rand = np.random.randn(100)
colors = np.random.rand(100)
plt.scatter(x_rand, y_rand, c=colors, alpha=0.6, cmap='viridis')
plt.title('随机散点图')
plt.colorbar(label='颜色值')

# 子图 3: 柱状图
plt.subplot(2, 2, 3)
categories = ['A', 'B', 'C', 'D', 'E']
values = [3, 7, 2, 5, 8]
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
plt.bar(categories, values, color=colors_bar)
plt.title('柱状图')
plt.xlabel('类别')
plt.ylabel('数值')

# 子图 4: 直方图
plt.subplot(2, 2, 4)
data = np.random.randn(1000)
plt.hist(data, bins=30, color='#6C5CE7', alpha=0.7, edgecolor='black')
plt.title('正态分布直方图')
plt.xlabel('值')
plt.ylabel('频数')

plt.tight_layout()
plt.savefig('matplotlib_demo.png', dpi=300, bbox_inches='tight')
print("图形已保存为 matplotlib_demo.png")
plt.show()
