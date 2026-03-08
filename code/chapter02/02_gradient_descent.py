"""
第2章代码示例 02：梯度下降算法
可视化梯度下降过程
"""
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# 目标函数
# ================================

def f(x):
    """目标函数: f(x) = x^2 + 2x + 1"""
    return x ** 2 + 2 * x + 1

def df(x):
    """导数: f'(x) = 2x + 2"""
    return 2 * x + 2

# ================================
# 梯度下降算法
# ================================

def gradient_descent(start_x, learning_rate=0.1, iterations=50):
    """
    梯度下降算法
    
    参数:
        start_x: 起始点
        learning_rate: 学习率
        iterations: 迭代次数
    
    返回:
        历史路径
    """
    x = start_x
    history = [x]
    
    for i in range(iterations):
        gradient = df(x)  # 计算梯度
        x = x - learning_rate * gradient  # 更新 x
        history.append(x)
    
    return np.array(history)

# ================================
# 运行梯度下降
# ================================

start_x = -3
history = gradient_descent(start_x, learning_rate=0.1, iterations=20)

# ================================
# 可视化
# ================================

x = np.linspace(-4, 2, 100)
y = f(x)

plt.figure(figsize=(12, 5))

# 函数和下降路径
plt.subplot(1, 2, 1)
plt.plot(x, y, label='f(x) = x² + 2x + 1', linewidth=2)
plt.plot(history, f(history), 'ro-', markersize=5, label='梯度下降路径', linewidth=1.5)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('梯度下降过程', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 收敛曲线
plt.subplot(1, 2, 2)
plt.plot(history, 'o-', linewidth=2, markersize=5)
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('x 值', fontsize=12)
plt.title('x 值收敛过程', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent.png', dpi=300, bbox_inches='tight')
print("图形已保存为 gradient_descent.png")
plt.show()

# ================================
# 输出结果
# ================================

print(f"\n起点: x = {start_x}, f(x) = {f(start_x):.4f}")
print(f"终点: x = {history[-1]:.4f}, f(x) = {f(history[-1]):.4f}")
print(f"理论最小值: x = -1, f(x) = 0")
