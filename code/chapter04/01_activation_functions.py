"""
第4章代码示例：神经网络基础演示
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# 激活函数可视化
# ================================

def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh 激活函数"""
    return np.tanh(x)

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU 激活函数"""
    return np.where(x > 0, x, alpha * x)

# 绘制激活函数
x = np.linspace(-5, 5, 100)

plt.figure(figsize=(15, 8))

functions = [
    ('Sigmoid', sigmoid(x)),
    ('Tanh', tanh(x)),
    ('ReLU', relu(x)),
    ('Leaky ReLU', leaky_relu(x))
]

for i, (name, y) in enumerate(functions):
    plt.subplot(2, 2, i + 1)
    plt.plot(x, y, linewidth=2, color=f'C{i}')
    plt.title(f'{name} 激活函数', fontsize=12)
    plt.xlabel('x', fontsize=10)
    plt.ylabel('f(x)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
print("激活函数图已保存为 activation_functions.png")
plt.show()

# ================================
# 激活函数特性对比
# ================================

print("\n=== 激活函数特性 ===")
print(f"Sigmoid 输出范围: (0, 1)")
print(f"Tanh 输出范围: (-1, 1)")
print(f"ReLU 输出范围: [0, +∞)")
print(f"Leaky ReLU 输出范围: (-∞, +∞)")
