"""
第1章代码示例 02：NumPy 基础
NumPy 数组操作演示
"""
import numpy as np

# ================================
# 1. 数组创建
# ================================

print("=== 数组创建 ===")

# 从列表创建
arr1 = np.array([1, 2, 3, 4, 5])
print(f"一维数组: {arr1}")

# 二维数组
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"二维数组:\n{arr2}")

# 特殊数组
zeros = np.zeros((3, 3))
print(f"全零矩阵:\n{zeros}")

ones = np.ones((2, 4))
print(f"全一矩阵:\n{ones}")

# 随机数组
random_arr = np.random.randn(3, 3)
print(f"随机矩阵:\n{random_arr}")

print("\n" + "="*50 + "\n")

# ================================
# 2. 数组运算
# ================================

print("=== 数组运算 ===")

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"a ** 2 = {a ** 2}")

# 矩阵乘法
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"\n矩阵 A:\n{A}")
print(f"矩阵 B:\n{B}")
print(f"矩阵乘法 A @ B:\n{A @ B}")

print("\n" + "="*50 + "\n")

# ================================
# 3. 数组统计
# ================================

print("=== 数组统计 ===")

data = np.random.randn(1000)

print(f"数据个数: {len(data)}")
print(f"平均值: {np.mean(data):.4f}")
print(f"标准差: {np.std(data):.4f}")
print(f"最小值: {np.min(data):.4f}")
print(f"最大值: {np.max(data):.4f}")
print(f"中位数: {np.median(data):.4f}")

print("\n" + "="*50 + "\n")

# ================================
# 4. 数组索引与切片
# ================================

print("=== 数组索引与切片 ===")

arr = np.arange(12).reshape(3, 4)
print(f"原数组:\n{arr}")

print(f"\n第一行: {arr[0, :]}")
print(f"第一列: {arr[:, 0]}")
print(f"中心 2x2:\n{arr[1:3, 1:3]}")

# 布尔索引
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"\n原数组: {data}")
print(f"大于 5 的元素: {data[data > 5]}")
print(f"偶数: {data[data % 2 == 0]}")
