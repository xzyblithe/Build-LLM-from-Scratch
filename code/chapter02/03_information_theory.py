"""
第2章代码示例 03：信息论应用
计算熵、交叉熵和 KL 散度
"""
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# 1. 熵的计算
# ================================

def entropy(probabilities):
    """
    计算熵
    
    参数:
        probabilities: 概率分布数组
    
    返回:
        熵值（bits）
    """
    # 去除概率为0的项
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

# 示例
print("=== 熵的计算 ===")

p1 = np.array([0.5, 0.5])
print(f"均匀分布 {p1} 的熵: {entropy(p1):.4f} bits")

p2 = np.array([0.9, 0.1])
print(f"不均匀分布 {p2} 的熵: {entropy(p2):.4f} bits")

p3 = np.array([1.0, 0.0])
print(f"确定分布 {p3} 的熵: {entropy(p3):.4f} bits")

print("\n" + "="*50 + "\n")

# ================================
# 2. 交叉熵
# ================================

def cross_entropy(p_true, p_pred):
    """
    计算交叉熵
    
    参数:
        p_true: 真实分布
        p_pred: 预测分布
    
    返回:
        交叉熵值
    """
    epsilon = 1e-10
    p_pred = np.clip(p_pred, epsilon, 1 - epsilon)
    return -np.sum(p_true * np.log(p_pred))

print("=== 交叉熵计算 ===")

y_true = np.array([0, 0, 1, 0])
y_pred_good = np.array([0.1, 0.1, 0.7, 0.1])
y_pred_bad = np.array([0.6, 0.2, 0.1, 0.1])

print(f"真实标签: {y_true}")
print(f"好的预测: {y_pred_good}")
print(f"差的预测: {y_pred_bad}")
print(f"\n好的预测交叉熵: {cross_entropy(y_true, y_pred_good):.4f}")
print(f"差的预测交叉熵: {cross_entropy(y_true, y_pred_bad):.4f}")

print("\n" + "="*50 + "\n")

# ================================
# 3. KL 散度
# ================================

def kl_divergence(p, q):
    """
    计算 KL 散度
    
    参数:
        p: 第一个概率分布
        q: 第二个概率分布
    
    返回:
        KL 散度值
    """
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1 - epsilon)
    q = np.clip(q, epsilon, 1 - epsilon)
    return np.sum(p * np.log(p / q))

print("=== KL 散度计算 ===")

p = np.array([0.2, 0.3, 0.5])
q1 = np.array([0.2, 0.3, 0.5])
q2 = np.array([0.3, 0.4, 0.3])

print(f"P = {p}")
print(f"Q1 = {q1}")
print(f"Q2 = {q2}")
print(f"\nKL(P||Q1) = {kl_divergence(p, q1):.4f}")
print(f"KL(P||Q2) = {kl_divergence(p, q2):.4f}")

print("\n" + "="*50 + "\n")

# ================================
# 4. 可视化
# ================================

# 不同概率分布的熵
probabilities = [
    np.array([0.5, 0.5]),
    np.array([0.6, 0.4]),
    np.array([0.7, 0.3]),
    np.array([0.8, 0.2]),
    np.array([0.9, 0.1]),
    np.array([0.95, 0.05]),
    np.array([0.99, 0.01]),
]

entropies = [entropy(p) for p in probabilities]
labels = [f'{p[0]:.2f}/{p[1]:.2f}' for p in probabilities]

plt.figure(figsize=(10, 5))
plt.bar(range(len(entropies)), entropies, color='skyblue', edgecolor='navy')
plt.xticks(range(len(entropies)), labels, rotation=45)
plt.xlabel('概率分布', fontsize=12)
plt.ylabel('熵（bits）', fontsize=12)
plt.title('不同概率分布的熵值', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')

for i, e in enumerate(entropies):
    plt.text(i, e + 0.02, f'{e:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('entropy_visualization.png', dpi=300, bbox_inches='tight')
print("图形已保存为 entropy_visualization.png")
plt.show()
