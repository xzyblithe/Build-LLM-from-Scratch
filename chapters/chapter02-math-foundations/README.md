# 第2章：数学基础（只讲必要的部分）

<div align="center">

[⬅️ 上一章](../chapter01-python-basics/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter03-ml-fundamentals/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解深度学习必备的线性代数概念
- ✅ 掌握概率统计的基本知识
- ✅ 理解梯度下降的数学原理
- ✅ 了解信息论在深度学习中的应用
- ✅ 用 Python 实现数学运算

---

## 🎯 本章内容

### 1. 线性代数基础

线性代数是深度学习的数学基础，我们将学习最核心的概念。

#### 1.1 向量（Vector）

**什么是向量？**

向量是既有大小又有方向的量。在深度学习中，我们用向量表示数据。

```python
import numpy as np

# 创建向量
v1 = np.array([1, 2, 3])  # 一维向量
v2 = np.array([4, 5, 6])

print(f"向量 v1: {v1}")
print(f"向量 v2: {v2}")
```

**向量运算**：

```python
# 向量加法
v_add = v1 + v2
print(f"v1 + v2 = {v_add}")

# 向量数乘
v_scalar = 2 * v1
print(f"2 * v1 = {v_scalar}")

# 向量点积（内积）
dot_product = np.dot(v1, v2)
print(f"v1 · v2 = {dot_product}")  # 1*4 + 2*5 + 3*6 = 32

# 向量长度（模）
norm = np.linalg.norm(v1)
print(f"|v1| = {norm:.4f}")

# 向量夹角余弦
cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print(f"cos(θ) = {cos_angle:.4f}")
```

**直观理解**：

```
向量 v1 = [1, 2, 3]
可以看作三维空间中的一个点 (1, 2, 3)
或者从原点指向该点的箭头

向量点积：
v1 · v2 = |v1| × |v2| × cos(θ)
表示两个向量的相似度
```

---

#### 1.2 矩阵（Matrix）

矩阵是二维数组，是深度学习中最常用的数据结构。

```python
import numpy as np

# 创建矩阵
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
])

print(f"矩阵 A (2x3):\n{A}")
print(f"\n矩阵 B (3x2):\n{B}")
```

**矩阵运算**：

```python
# 矩阵形状
print(f"A 的形状: {A.shape}")  # (2, 3)
print(f"B 的形状: {B.shape}")  # (3, 2)

# 矩阵加法（需要相同形状）
C = np.array([[1, 2, 3], [4, 5, 6]])
D = np.array([[7, 8, 9], [10, 11, 12]])
print(f"\n矩阵加法:\n{C + D}")

# 矩阵乘法（点积）
AB = np.dot(A, B)  # 或 A @ B
print(f"\n矩阵乘法 A × B (2x2):\n{AB}")

# 逐元素乘法（Hadamard 积）
E = np.array([[1, 2], [3, 4]])
F = np.array([[5, 6], [7, 8]])
print(f"\n逐元素乘法:\n{E * F}")
```

**矩阵乘法规则**：

```
矩阵 A: m × n
矩阵 B: n × p
结果 C: m × p

C[i,j] = Σ A[i,k] × B[k,j]

示例：
A (2×3) × B (3×2) = C (2×2)

C[0,0] = 1×7 + 2×9 + 3×11 = 58
C[0,1] = 1×8 + 2×10 + 3×12 = 64
C[1,0] = 4×7 + 5×9 + 6×11 = 139
C[1,1] = 4×8 + 5×10 + 6×12 = 154
```

**转置**：

```python
# 矩阵转置
A_T = A.T
print(f"A 的转置 (3x2):\n{A_T}")

# 转置的性质
print(f"\n(A^T)^T = A: {(A.T).T == A).all()}")
```

---

#### 1.3 矩阵的逆

只有方阵（行数=列数）才可能有逆矩阵。

```python
import numpy as np

# 创建方阵
M = np.array([
    [1, 2],
    [3, 4]
])

print(f"矩阵 M:\n{M}")

# 计算逆矩阵
M_inv = np.linalg.inv(M)
print(f"\n逆矩阵 M^(-1):\n{M_inv}")

# 验证：M × M^(-1) = I（单位矩阵）
I = M @ M_inv
print(f"\nM × M^(-1):\n{I}")
# 接近单位矩阵 [[1, 0], [0, 1]]
```

**逆矩阵的作用**：

在深度学习中，逆矩阵用于：
- 求解线性方程组
- 最小二乘法
- 优化问题

---

#### 1.4 特征值与特征向量

```python
import numpy as np

# 创建矩阵
A = np.array([
    [4, 2],
    [1, 3]
])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"特征值:\n{eigenvalues}")
print(f"\n特征向量（每列对应一个特征值）:\n{eigenvectors}")

# 验证: A × v = λ × v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lambda_v = eigenvalues[i]
    
    left = A @ v
    right = lambda_v * v
    
    print(f"\n特征值 λ{i+1} = {lambda_v:.4f}")
    print(f"A × v = {left}")
    print(f"λ × v = {right}")
```

---

### 2. 概率与统计基础

#### 2.1 概率基础

**基本概念**：

```python
import numpy as np

# 抛硬币实验
def coin_flip_experiment(n=10000):
    """模拟抛硬币实验"""
    flips = np.random.choice(['H', 'T'], size=n, p=[0.5, 0.5])
    heads = np.sum(flips == 'H')
    prob_heads = heads / n
    return prob_heads

print(f"理论概率: 0.5")
print(f"实验概率: {coin_flip_experiment(10000):.4f}")
```

**条件概率**：

```
P(A|B) = P(A∩B) / P(B)

在 B 发生的条件下 A 发生的概率
```

**贝叶斯定理**：

```
P(A|B) = P(B|A) × P(A) / P(B)

后验概率 = 似然 × 先验 / 证据
```

```python
# 贝叶斯定理示例：疾病诊断
"""
已知：
- P(病) = 0.01（患病率）
- P(阳性|病) = 0.99（真阳性率）
- P(阳性|健康) = 0.05（假阳性率）

求：P(病|阳性)
"""

P_disease = 0.01
P_positive_given_disease = 0.99
P_positive_given_healthy = 0.05

# P(阳性) = P(阳性|病)P(病) + P(阳性|健康)P(健康)
P_positive = (P_positive_given_disease * P_disease + 
              P_positive_given_healthy * (1 - P_disease))

# 贝叶斯定理
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print(f"检测阳性时，实际患病的概率: {P_disease_given_positive:.4f}")
# 结果约为 16.7%，远低于直觉！
```

---

#### 2.2 常见概率分布

**均匀分布**：

```python
import numpy as np
import matplotlib.pyplot as plt

# 均匀分布
samples = np.random.uniform(0, 1, 10000)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(samples, bins=50, density=True, alpha=0.7)
plt.title('均匀分布')
plt.xlabel('值')
plt.ylabel('概率密度')

plt.tight_layout()
plt.show()
```

**正态分布（高斯分布）**：

```python
import numpy as np
import matplotlib.pyplot as plt

# 正态分布
samples = np.random.normal(loc=0, scale=1, size=10000)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(samples, bins=50, density=True, alpha=0.7)
plt.title('正态分布')
plt.xlabel('值')
plt.ylabel('概率密度')

# 理论曲线
from scipy.stats import norm
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, 0, 1), 'r-', linewidth=2, label='理论曲线')
plt.legend()

plt.tight_layout()
plt.show()
```

**标准差的意义**：

```
正态分布 N(μ, σ²)
- μ: 均值（中心位置）
- σ: 标准差（分散程度）

68-95-99.7 规则：
- 68% 的数据在 μ±1σ 内
- 95% 的数据在 μ±2σ 内
- 99.7% 的数据在 μ±3σ 内
```

---

### 3. 微积分基础

#### 3.1 导数的直观理解

导数表示函数在某点的变化率。

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def f(x):
    return x ** 2

# 定义导数
def df(x):
    return 2 * x

# 绘制函数和导数
x = np.linspace(-3, 3, 100)
y = f(x)

plt.figure(figsize=(12, 5))

# 函数曲线
plt.subplot(1, 2, 1)
plt.plot(x, y, label='f(x) = x²')
plt.title('函数 f(x) = x²')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)
plt.legend()

# 切线（导数的几何意义）
x0 = 2  # 切点
y0 = f(x0)
slope = df(x0)
tangent_y = slope * (x - x0) + y0

plt.plot(x, tangent_y, 'r--', label=f'切线 (斜率={slope})')
plt.plot(x0, y0, 'ro', markersize=10, label='切点')
plt.legend()

# 导数曲线
plt.subplot(1, 2, 2)
plt.plot(x, df(x), label="f'(x) = 2x", color='orange')
plt.title("导数 f'(x) = 2x")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
```

**常用导数公式**：

```
d/dx (x^n) = n × x^(n-1)
d/dx (e^x) = e^x
d/dx (ln x) = 1/x
d/dx (sin x) = cos x
d/dx (cos x) = -sin x
```

---

#### 3.2 梯度下降

梯度下降是深度学习的核心优化算法。

```python
import numpy as np
import matplotlib.pyplot as plt

# 目标函数
def f(x):
    return x ** 2 + 2 * x + 1

# 导数（梯度）
def df(x):
    return 2 * x + 2

# 梯度下降
def gradient_descent(start_x, learning_rate=0.1, iterations=50):
    """梯度下降算法"""
    x = start_x
    history = [x]
    
    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        history.append(x)
    
    return np.array(history)

# 运行梯度下降
start_x = -3
history = gradient_descent(start_x, learning_rate=0.1, iterations=20)

# 可视化
x = np.linspace(-4, 2, 100)
y = f(x)

plt.figure(figsize=(12, 5))

# 函数和下降路径
plt.subplot(1, 2, 1)
plt.plot(x, y, label='f(x)')
plt.plot(history, f(history), 'ro-', markersize=5, label='梯度下降路径')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('梯度下降过程')
plt.legend()
plt.grid(True, alpha=0.3)

# 收敛曲线
plt.subplot(1, 2, 2)
plt.plot(history, 'o-')
plt.xlabel('迭代次数')
plt.ylabel('x 值')
plt.title('x 值收敛过程')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"起点: x = {start_x}, f(x) = {f(start_x):.4f}")
print(f"终点: x = {history[-1]:.4f}, f(x) = {f(history[-1]):.4f}")
print(f"理论最小值: x = -1, f(x) = 0")
```

**梯度下降原理**：

```
梯度：函数上升最快的方向
负梯度：函数下降最快的方向

更新公式：
x_new = x_old - α × ∇f(x_old)

α: 学习率（步长）
∇f: 梯度（导数）
```

**学习率的影响**：

```python
import numpy as np
import matplotlib.pyplot as plt

# 不同学习率对比
learning_rates = [0.01, 0.1, 0.5, 1.0]
iterations = 20

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    history = gradient_descent(-3, learning_rate=lr, iterations=iterations)
    plt.plot(history, 'o-', label=f'lr={lr}', markersize=4)

plt.xlabel('迭代次数')
plt.ylabel('x 值')
plt.title('不同学习率的收敛过程')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

#### 3.3 链式法则

链式法则是反向传播算法的数学基础。

```
如果 y = f(u) 且 u = g(x)
则 dy/dx = dy/du × du/dx

链式法则用于复合函数求导
```

```python
import numpy as np

# 复合函数示例
def g(x):
    """u = g(x) = x^2"""
    return x ** 2

def f(u):
    """y = f(u) = e^u"""
    return np.exp(u)

def composite_function(x):
    """y = f(g(x)) = e^(x^2)"""
    return f(g(x))

# 数值梯度
def numerical_derivative(func, x, h=1e-5):
    """数值求导"""
    return (func(x + h) - func(x - h)) / (2 * h)

# 链式法则求导
def chain_rule_derivative(x):
    """使用链式法则求导"""
    # u = x^2, du/dx = 2x
    u = g(x)
    du_dx = 2 * x
    
    # y = e^u, dy/du = e^u
    dy_du = np.exp(u)
    
    # dy/dx = dy/du × du/dx
    dy_dx = dy_du * du_dx
    
    return dy_dx

# 测试
x = 2.0

print(f"x = {x}")
print(f"u = g(x) = {g(x)}")
print(f"y = f(u) = {f(g(x)):.4f}")
print(f"\n数值导数: {numerical_derivative(composite_function, x):.4f}")
print(f"链式法则导数: {chain_rule_derivative(x):.4f}")
```

---

### 4. 信息论基础

#### 4.1 熵（Entropy）

熵衡量信息的不确定性。

```
H(X) = -Σ P(x) × log P(x)

熵越大，不确定性越大
```

```python
import numpy as np

def entropy(probabilities):
    """计算熵"""
    # 去除概率为0的项（避免 log(0)）
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

# 示例 1：均匀分布（最大熵）
p1 = np.array([0.5, 0.5])
print(f"均匀分布 {p1} 的熵: {entropy(p1):.4f} bits")

# 示例 2：不均匀分布
p2 = np.array([0.9, 0.1])
print(f"不均匀分布 {p2} 的熵: {entropy(p2):.4f} bits")

# 示例 3：确定分布（最小熵）
p3 = np.array([1.0, 0.0])
print(f"确定分布 {p3} 的熵: {entropy(p3):.4f} bits")
```

---

#### 4.2 交叉熵（Cross-Entropy）

交叉熵用于衡量两个概率分布的差异，是深度学习中常用的损失函数。

```
H(P, Q) = -Σ P(x) × log Q(x)

P: 真实分布
Q: 预测分布
```

```python
import numpy as np

def cross_entropy(p_true, p_pred):
    """计算交叉熵"""
    # 避免数值问题
    epsilon = 1e-10
    p_pred = np.clip(p_pred, epsilon, 1 - epsilon)
    return -np.sum(p_true * np.log(p_pred))

# 分类任务示例
# 真实标签：类别 2（one-hot 编码）
y_true = np.array([0, 0, 1, 0])

# 预测概率
y_pred_good = np.array([0.1, 0.1, 0.7, 0.1])  # 预测正确
y_pred_bad = np.array([0.6, 0.2, 0.1, 0.1])   # 预测错误

print(f"好的预测交叉熵: {cross_entropy(y_true, y_pred_good):.4f}")
print(f"差的预测交叉熵: {cross_entropy(y_true, y_pred_bad):.4f}")
```

---

#### 4.3 KL 散度（KL Divergence）

KL 散度衡量两个概率分布的相似度。

```
KL(P||Q) = Σ P(x) × log(P(x)/Q(x))

KL 散度 ≥ 0
KL 散度 = 0 当且仅当 P = Q
```

```python
import numpy as np

def kl_divergence(p, q):
    """计算 KL 散度"""
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1 - epsilon)
    q = np.clip(q, epsilon, 1 - epsilon)
    return np.sum(p * np.log(p / q))

# 示例
p = np.array([0.2, 0.3, 0.5])
q1 = np.array([0.2, 0.3, 0.5])  # 相同分布
q2 = np.array([0.3, 0.4, 0.3])  # 不同分布

print(f"P 和 P 的 KL 散度: {kl_divergence(p, q1):.4f}")
print(f"P 和 Q 的 KL 散度: {kl_divergence(p, q2):.4f}")
```

---

## 💻 完整代码示例

### 示例 1：矩阵运算库

```python
"""
示例 1：从零实现矩阵运算库
"""
import numpy as np

class Matrix:
    """矩阵类"""
    
    def __init__(self, data):
        """初始化矩阵"""
        self.data = np.array(data)
        self.shape = self.data.shape
    
    def __repr__(self):
        return f"Matrix({self.data})"
    
    def __add__(self, other):
        """矩阵加法"""
        if self.shape != other.shape:
            raise ValueError("矩阵形状不匹配")
        return Matrix(self.data + other.data)
    
    def __mul__(self, other):
        """矩阵乘法"""
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("矩阵形状不匹配")
            return Matrix(self.data @ other.data)
        else:
            return Matrix(self.data * other)
    
    def transpose(self):
        """转置"""
        return Matrix(self.data.T)
    
    def determinant(self):
        """行列式（仅方阵）"""
        if self.shape[0] != self.shape[1]:
            raise ValueError("非方阵")
        return np.linalg.det(self.data)
    
    def inverse(self):
        """逆矩阵（仅方阵）"""
        if self.shape[0] != self.shape[1]:
            raise ValueError("非方阵")
        return Matrix(np.linalg.inv(self.data))
    
    def eigen(self):
        """特征值和特征向量"""
        eigenvalues, eigenvectors = np.linalg.eig(self.data)
        return eigenvalues, eigenvectors

# 测试
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print(f"矩阵 A: {A}")
print(f"矩阵 B: {B}")
print(f"\nA + B: {A + B}")
print(f"\nA × B: {A * B}")
print(f"\nA 的转置: {A.transpose()}")
print(f"\nA 的行列式: {A.determinant():.2f}")
print(f"\nA 的逆矩阵: {A.inverse()}")

eigenvalues, eigenvectors = A.eigen()
print(f"\nA 的特征值: {eigenvalues}")
print(f"A 的特征向量:\n{eigenvectors}")
```

---

### 示例 2：梯度下降可视化

```python
"""
示例 2：梯度下降算法可视化
"""
import numpy as np
import matplotlib.pyplot as plt

# 目标函数：二元函数
def f(x, y):
    """f(x, y) = x^2 + y^2"""
    return x ** 2 + y ** 2

# 梯度
def gradient(x, y):
    """梯度 [∂f/∂x, ∂f/∂y]"""
    return np.array([2 * x, 2 * y])

# 梯度下降
def gradient_descent_2d(start_point, learning_rate=0.1, iterations=50):
    """二维梯度下降"""
    path = [start_point.copy()]
    point = start_point.copy()
    
    for _ in range(iterations):
        grad = gradient(point[0], point[1])
        point = point - learning_rate * grad
        path.append(point.copy())
    
    return np.array(path)

# 运行
start_point = np.array([3.0, 3.0])
path = gradient_descent_2d(start_point, learning_rate=0.1, iterations=30)

# 可视化
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(12, 5))

# 等高线图
plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=4, label='梯度下降路径')
plt.xlabel('x')
plt.ylabel('y')
plt.title('梯度下降路径（等高线）')
plt.legend()
plt.grid(True, alpha=0.3)

# 3D 图
ax = plt.subplot(1, 2, 2, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
ax.plot(path[:, 0], path[:, 1], f(path[:, 0], path[:, 1]), 'ro-', linewidth=2, markersize=4)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('梯度下降路径（3D）')

plt.tight_layout()
plt.show()

print(f"起点: {start_point}, f(x, y) = {f(*start_point):.4f}")
print(f"终点: {path[-1]}, f(x, y) = {f(*path[-1]):.4f}")
```

---

### 示例 3：信息论应用

```python
"""
示例 3：信息论在文本分析中的应用
"""
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def calculate_entropy(text):
    """计算文本的熵"""
    # 统计字符频率
    counter = Counter(text)
    total = len(text)
    probabilities = np.array([count / total for count in counter.values()])
    
    # 计算熵
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy, counter

# 示例文本
texts = {
    '重复文本': 'AAAAAABBBBBBCCCCCC',
    '随机文本': 'ABCDEFGHJKLMNOPQR',
    '英文文本': 'The quick brown fox jumps over the lazy dog'
}

# 计算熵
results = {}
for name, text in texts.items():
    entropy, counter = calculate_entropy(text)
    results[name] = {
        'entropy': entropy,
        'unique_chars': len(counter),
        'length': len(text)
    }

# 可视化
plt.figure(figsize=(10, 5))

names = list(results.keys())
entropies = [results[name]['entropy'] for name in names]

plt.bar(names, entropies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.ylabel('熵（bits）')
plt.title('不同文本的熵值对比')
plt.grid(True, alpha=0.3, axis='y')

for i, (name, entropy) in enumerate(zip(names, entropies)):
    plt.text(i, entropy + 0.05, f'{entropy:.3f}', ha='center', fontsize=12)

plt.tight_layout()
plt.show()

# 打印结果
print("\n=== 文本熵分析 ===")
for name, data in results.items():
    print(f"\n{name}:")
    print(f"  文本长度: {data['length']}")
    print(f"  唯一字符数: {data['unique_chars']}")
    print(f"  熵值: {data['entropy']:.4f} bits")
```

---

## 🎯 实践练习

### 练习 1：实现矩阵乘法

**任务**：不使用 NumPy，手写实现矩阵乘法。

**要求**：
- 实现 `matrix_multiply(A, B)` 函数
- 检查矩阵形状是否匹配
- 返回结果矩阵

**参考代码**：

```python
def matrix_multiply(A, B):
    """
    手写实现矩阵乘法
    
    参数:
        A: m×n 矩阵
        B: n×p 矩阵
    
    返回:
        C: m×p 矩阵
    """
    # 检查形状
    if len(A[0]) != len(B):
        raise ValueError("矩阵形状不匹配")
    
    m, n, p = len(A), len(A[0]), len(B[0])
    
    # 初始化结果矩阵
    C = [[0 for _ in range(p)] for _ in range(m)]
    
    # 矩阵乘法
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C

# 测试
A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 8], [9, 10], [11, 12]]

result = matrix_multiply(A, B)
print("A × B =")
for row in result:
    print(row)
```

---

### 练习 2：实现梯度下降

**任务**：实现梯度下降算法优化二元函数。

**函数**：`f(x, y) = x^2 + 2y^2 - xy + x - y`

**要求**：
- 求出函数的梯度
- 实现梯度下降算法
- 找到最小值点

**参考代码**：

```python
import numpy as np

def f(x, y):
    """目标函数"""
    return x**2 + 2*y**2 - x*y + x - y

def gradient(x, y):
    """梯度"""
    df_dx = 2*x - y + 1
    df_dy = 4*y - x - 1
    return np.array([df_dx, df_dy])

def gradient_descent(start, learning_rate=0.1, iterations=100):
    """梯度下降"""
    point = np.array(start, dtype=float)
    path = [point.copy()]
    
    for _ in range(iterations):
        grad = gradient(point[0], point[1])
        point = point - learning_rate * grad
        path.append(point.copy())
    
    return np.array(path)

# 运行
start = [2, 2]
path = gradient_descent(start, learning_rate=0.1, iterations=50)

print(f"起点: ({start[0]}, {start[1]}), f = {f(*start):.4f}")
print(f"终点: ({path[-1, 0]:.4f}, {path[-1, 1]:.4f}), f = {f(*path[-1]):.4f}")

# 理论最小值点
# 解方程组：2x - y + 1 = 0, -x + 4y - 1 = 0
# 得 x = -3/7, y = 1/7
print(f"\n理论最小值点: ({-3/7:.4f}, {1/7:.4f})")
```

---

## 📝 本章小结

### 核心要点

1. **线性代数**：向量、矩阵、矩阵乘法、逆矩阵、特征值
2. **概率统计**：概率分布、贝叶斯定理、期望、方差
3. **微积分**：导数、梯度下降、链式法则
4. **信息论**：熵、交叉熵、KL 散度

### 数学在深度学习中的应用

| 数学概念 | 深度学习应用 |
|----------|-------------|
| 矩阵乘法 | 神经网络前向传播 |
| 梯度下降 | 模型优化 |
| 链式法则 | 反向传播 |
| 交叉熵 | 损失函数 |
| 概率分布 | 采样、生成模型 |

### 下一步

- 完成实践练习
- 理解梯度下降的数学原理
- 准备进入第 3 章：机器学习基础

---

<div align="center">

[⬅️ 上一章](../chapter01-python-basics/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter03-ml-fundamentals/README.md)

</div>
