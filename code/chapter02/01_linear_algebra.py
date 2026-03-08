"""
第2章代码示例 01：线性代数基础
向量、矩阵运算演示
"""
import numpy as np

# ================================
# 1. 向量运算
# ================================

print("=== 向量运算 ===")

# 创建向量
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"向量 v1: {v1}")
print(f"向量 v2: {v2}")

# 向量加法
print(f"\nv1 + v2 = {v1 + v2}")

# 向量数乘
print(f"2 * v1 = {2 * v1}")

# 向量点积
dot_product = np.dot(v1, v2)
print(f"v1 · v2 = {dot_product}")

# 向量长度
norm = np.linalg.norm(v1)
print(f"|v1| = {norm:.4f}")

# 向量夹角余弦
cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(f"cos(θ) = {cos_angle:.4f}")

print("\n" + "="*50 + "\n")

# ================================
# 2. 矩阵运算
# ================================

print("=== 矩阵运算 ===")

# 创建矩阵
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])

print(f"矩阵 A (2x3):\n{A}")
print(f"\n矩阵 B (3x2):\n{B}")

# 矩阵形状
print(f"\nA 的形状: {A.shape}")
print(f"B 的形状: {B.shape}")

# 矩阵乘法
AB = np.dot(A, B)
print(f"\n矩阵乘法 A × B (2x2):\n{AB}")

# 转置
print(f"\nA 的转置 (3x2):\n{A.T}")

print("\n" + "="*50 + "\n")

# ================================
# 3. 矩阵求逆
# ================================

print("=== 矩阵求逆 ===")

M = np.array([[1, 2], [3, 4]])
print(f"矩阵 M:\n{M}")

M_inv = np.linalg.inv(M)
print(f"\n逆矩阵 M^(-1):\n{M_inv}")

# 验证
I = M @ M_inv
print(f"\nM × M^(-1):\n{I}")

print("\n" + "="*50 + "\n")

# ================================
# 4. 特征值与特征向量
# ================================

print("=== 特征值与特征向量 ===")

A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"矩阵 A:\n{A}")
print(f"\n特征值: {eigenvalues}")
print(f"\n特征向量:\n{eigenvectors}")

# 验证
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_v = eigenvalues[i]
    
    print(f"\n特征值 λ{i+1} = {lambda_v:.4f}")
    print(f"A × v = {A @ v}")
    print(f"λ × v = {lambda_v * v}")
