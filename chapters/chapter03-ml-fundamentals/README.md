# 第3章：机器学习基础概念

<div align="center">

[⬅️ 上一章](../chapter02-math-foundations/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter04-neural-networks/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解机器学习的基本概念和分类
- ✅ 掌握监督学习、无监督学习、强化学习的区别
- ✅ 理解训练集、验证集、测试集的作用
- ✅ 掌握过拟合与欠拟合的原因和解决方案
- ✅ 理解损失函数、优化器、学习率等核心概念

---

## 🎯 本章内容

### 1. 什么是机器学习？

#### 1.1 机器学习的定义

**机器学习**：让计算机从数据中学习规律，而不是明确编程。

```
传统编程：规则 + 数据 → 结果
机器学习：数据 + 结果 → 规则
```

**核心思想**：

```python
# 传统编程：我们告诉计算机如何做
def classify_email(email):
    if "优惠" in email or "促销" in email:
        return "垃圾邮件"
    else:
        return "正常邮件"

# 机器学习：计算机自己学会如何做
# 输入：大量标注好的邮件数据
# 输出：自动学会分类规则
```

---

#### 1.2 机器学习的分类

| 类型 | 特点 | 应用场景 |
|------|------|----------|
| **监督学习** | 有标签数据 | 分类、回归 |
| **无监督学习** | 无标签数据 | 聚类、降维 |
| **强化学习** | 奖励机制 | 游戏、机器人 |
| **半监督学习** | 少量标签 | 标注成本高的场景 |

**监督学习示例**：

```python
# 监督学习：垃圾邮件分类
# 输入：邮件内容 + 标签（垃圾/正常）
emails = [
    ("恭喜您中奖了！", "垃圾"),
    ("明天开会讨论项目", "正常"),
    ("限时优惠，点击领取", "垃圾"),
    ("项目进度汇报", "正常"),
]

# 模型学习邮件内容和标签之间的关系
```

**无监督学习示例**：

```python
# 无监督学习：客户聚类
# 输入：客户数据（无标签）
customers = [
    [25, 3000],  # [年龄, 收入]
    [30, 5000],
    [28, 4500],
    [50, 15000],
    [55, 18000],
]

# 模型自动发现客户群体
```

---

### 2. 数据集划分

#### 2.1 训练集、验证集、测试集

```python
import numpy as np
from sklearn.model_selection import train_test_split

# 生成示例数据
X = np.random.randn(1000, 10)  # 1000个样本，每个10个特征
y = np.random.randint(0, 2, 1000)  # 二分类标签

# 第一次划分：训练集 + 临时集
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 第二次划分：验证集 + 测试集
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"验证集: {X_val.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")
```

**各数据集的作用**：

| 数据集 | 作用 | 比例 |
|--------|------|------|
| **训练集** | 训练模型，学习参数 | 60-80% |
| **验证集** | 调整超参数，选择模型 | 10-20% |
| **测试集** | 评估最终性能 | 10-20% |

---

#### 2.2 交叉验证

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# 创建模型
model = LogisticRegression()

# 5折交叉验证
scores = cross_val_score(model, X, y, cv=5)

print(f"交叉验证分数: {scores}")
print(f"平均分数: {scores.mean():.4f}")
print(f"标准差: {scores.std():.4f}")
```

**交叉验证流程**：

```
数据集: [1, 2, 3, 4, 5]

第1折: 训练[2,3,4,5] 验证[1]
第2折: 训练[1,3,4,5] 验证[2]
第3折: 训练[1,2,4,5] 验证[3]
第4折: 训练[1,2,3,5] 验证[4]
第5折: 训练[1,2,3,4] 验证[5]

最终分数 = 平均(各折分数)
```

---

### 3. 过拟合与欠拟合

#### 3.1 概念理解

```
欠拟合（Underfitting）：
- 模型太简单，无法捕捉数据规律
- 训练误差和测试误差都很高
- 解决：增加模型复杂度

过拟合（Overfitting）：
- 模型太复杂，记住了训练数据的噪声
- 训练误差低，测试误差高
- 解决：正则化、dropout、早停
```

**可视化理解**：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 生成数据
np.random.seed(42)
X = np.linspace(0, 1, 20)
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, 20)

# 不同复杂度的模型
degrees = [1, 3, 15]  # 欠拟合、适中、过拟合

plt.figure(figsize=(15, 4))

for i, degree in enumerate(degrees):
    plt.subplot(1, 3, i + 1)
    
    # 创建多项式回归模型
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # 训练
    model.fit(X.reshape(-1, 1), y)
    
    # 预测
    X_test = np.linspace(0, 1, 100)
    y_pred = model.predict(X_test.reshape(-1, 1))
    
    # 绘图
    plt.scatter(X, y, color='blue', s=50, label='训练数据')
    plt.plot(X_test, y_pred, 'r-', linewidth=2, label='模型拟合')
    plt.plot(X_test, np.sin(2 * np.pi * X_test), 'g--', label='真实函数')
    
    title = f'多项式次数={degree}'
    if degree == 1:
        title += ' (欠拟合)'
    elif degree == 3:
        title += ' (适中)'
    else:
        title += ' (过拟合)'
    
    plt.title(title, fontsize=12)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

#### 3.2 解决方案

**正则化**：

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso

# Ridge 正则化（L2）
ridge = Ridge(alpha=1.0)  # alpha 越大，正则化越强

# Lasso 正则化（L1）
lasso = Lasso(alpha=0.1)  # 可以产生稀疏解

# 对比
models = {
    '无正则化': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.1)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"{name}: 训练={train_score:.4f}, 测试={test_score:.4f}")
```

**早停（Early Stopping）**：

```python
import torch
import torch.nn as nn
import torch.optim as optim

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

# 使用示例
early_stopping = EarlyStopping(patience=5)

for epoch in range(100):
    # 训练模型
    train_loss = train_one_epoch(model, train_loader, optimizer)
    
    # 验证
    val_loss = validate(model, val_loader)
    
    # 检查是否早停
    if early_stopping(val_loss):
        print(f"早停于第 {epoch} 轮")
        break
```

---

### 4. 核心概念详解

#### 4.1 损失函数

损失函数衡量模型预测与真实值的差距。

**回归损失**：

```python
import numpy as np

# 均方误差（MSE）
def mse_loss(y_true, y_pred):
    """均方误差"""
    return np.mean((y_true - y_pred) ** 2)

# 平均绝对误差（MAE）
def mae_loss(y_true, y_pred):
    """平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))

# 示例
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

print(f"MSE: {mse_loss(y_true, y_pred):.4f}")
print(f"MAE: {mae_loss(y_true, y_pred):.4f}")
```

**分类损失**：

```python
import numpy as np

# 交叉熵损失
def cross_entropy_loss(y_true, y_pred):
    """交叉熵损失"""
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# 示例
y_true = np.array([1, 0, 1, 0, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.7])

print(f"交叉熵损失: {cross_entropy_loss(y_true, y_pred):.4f}")
```

---

#### 4.2 优化器

优化器根据梯度更新模型参数。

**梯度下降的变体**：

```python
import numpy as np

class SGDOptimizer:
    """随机梯度下降"""
    
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def update(self, params, grads):
        """更新参数"""
        return params - self.lr * grads

class MomentumOptimizer:
    """带动量的梯度下降"""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params, grads):
        """更新参数"""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        self.velocity = self.momentum * self.velocity - self.lr * grads
        return params + self.velocity

class AdamOptimizer:
    """Adam 优化器"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 一阶矩
        self.v = None  # 二阶矩
        self.t = 0     # 时间步
    
    def update(self, params, grads):
        """更新参数"""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        
        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# 对比不同优化器
def compare_optimizers():
    """比较不同优化器"""
    # 目标函数：f(x) = x^2
    # 最小值点：x = 0
    
    start_x = 5.0
    iterations = 20
    
    optimizers = {
        'SGD': SGDOptimizer(learning_rate=0.1),
        'Momentum': MomentumOptimizer(learning_rate=0.1, momentum=0.9),
        'Adam': AdamOptimizer(learning_rate=0.5)
    }
    
    plt.figure(figsize=(10, 6))
    
    for name, optimizer in optimizers.items():
        x = start_x
        history = [x]
        
        for _ in range(iterations):
            grad = 2 * x  # f'(x) = 2x
            x = optimizer.update(x, grad)
            history.append(x)
        
        plt.plot(history, 'o-', label=name, markersize=4)
    
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('x 值', fontsize=12)
    plt.title('不同优化器收敛对比', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

compare_optimizers()
```

---

#### 4.3 学习率

学习率控制参数更新的步长。

```python
import numpy as np
import matplotlib.pyplot as plt

def learning_rate_effect():
    """学习率的影响"""
    
    # 目标函数
    def f(x):
        return x ** 2
    
    def df(x):
        return 2 * x
    
    learning_rates = [0.01, 0.1, 1.0, 2.0]
    start_x = 2.0
    iterations = 20
    
    plt.figure(figsize=(12, 8))
    
    for i, lr in enumerate(learning_rates):
        plt.subplot(2, 2, i + 1)
        
        x = start_x
        history = [x]
        
        for _ in range(iterations):
            x = x - lr * df(x)
            history.append(x)
        
        # 绘制函数曲线
        x_range = np.linspace(-3, 3, 100)
        plt.plot(x_range, f(x_range), 'b-', label='f(x) = x²')
        
        # 绘制优化路径
        plt.plot(history, f(np.array(history)), 'ro-', markersize=4, label='优化路径')
        
        plt.title(f'学习率 = {lr}', fontsize=12)
        plt.xlabel('x', fontsize=10)
        plt.ylabel('f(x)', fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 标注结果
        if lr < 0.1:
            result = "收敛慢"
        elif lr < 1.0:
            result = "收敛快"
        else:
            result = "发散" if abs(history[-1]) > 10 else "震荡"
        
        plt.text(0.05, 0.95, result, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

learning_rate_effect()
```

**学习率调度**：

```python
# 学习率衰减策略
def learning_rate_schedules():
    """不同的学习率调度策略"""
    
    epochs = 100
    initial_lr = 0.1
    
    # 1. 固定学习率
    fixed_lr = [initial_lr] * epochs
    
    # 2. 阶梯衰减
    step_lr = [initial_lr * (0.1 ** (i // 30)) for i in range(epochs)]
    
    # 3. 指数衰减
    exponential_lr = [initial_lr * (0.95 ** i) for i in range(epochs)]
    
    # 4. 余弦退火
    import math
    cosine_lr = [initial_lr * (0.5 * (1 + math.cos(i * math.pi / epochs))) 
                 for i in range(epochs)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(fixed_lr, label='固定学习率', linewidth=2)
    plt.plot(step_lr, label='阶梯衰减', linewidth=2)
    plt.plot(exponential_lr, label='指数衰减', linewidth=2)
    plt.plot(cosine_lr, label='余弦退火', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('学习率', fontsize=12)
    plt.title('不同的学习率调度策略', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

learning_rate_schedules()
```

---

### 5. 模型评估

#### 5.1 分类模型评估

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import numpy as np

# 示例数据
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])

# 计算各项指标
print(f"准确率 (Accuracy): {accuracy_score(y_true, y_pred):.4f}")
print(f"精确率 (Precision): {precision_score(y_true, y_pred):.4f}")
print(f"召回率 (Recall): {recall_score(y_true, y_pred):.4f}")
print(f"F1 分数: {f1_score(y_true, y_pred):.4f}")

# 混淆矩阵
print("\n混淆矩阵:")
print(confusion_matrix(y_true, y_pred))

# 详细报告
print("\n分类报告:")
print(classification_report(y_true, y_pred))
```

**混淆矩阵可视化**：

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 可视化
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.title('混淆矩阵', fontsize=14)
plt.show()
```

---

#### 5.2 回归模型评估

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np

# 示例数据
y_true = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
y_pred = np.array([2.5, 5.5, 6.8, 9.2, 10.8])

# 计算指标
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
```

---

## 💻 完整代码示例

### 示例 1：从零实现线性回归

```python
"""
示例 1：从零实现线性回归
"""
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """线性回归模型"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.n_iterations):
            # 预测
            y_pred = np.dot(X, self.weights) + self.bias
            
            # 计算损失
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # 计算梯度
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        """预测"""
        return np.dot(X, self.weights) + self.bias

# 测试
np.random.seed(42)
X = np.random.randn(100, 1) * 2
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.5

# 训练
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# 预测
X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.5, label='训练数据')
plt.plot(X_test, y_pred, 'r-', linewidth=2, label='预测线')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('线性回归拟合', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(model.loss_history)
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('MSE 损失', fontsize=12)
plt.title('损失下降曲线', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"学习到的参数: w = {model.weights[0]:.4f}, b = {model.bias:.4f}")
print(f"真实参数: w = 3.0, b = 2.0")
```

---

### 示例 2：K-近邻算法实现

```python
"""
示例 2：K-近邻分类算法
"""
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class KNNClassifier:
    """K-近邻分类器"""
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """训练（只是存储数据）"""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """预测"""
        predictions = []
        
        for x in X:
            # 计算距离
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # 找到最近的 k 个邻居
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            
            # 多数投票
            most_common = Counter(k_labels).most_common(1)
            predictions.append(most_common[0][0])
        
        return np.array(predictions)

# 测试
np.random.seed(42)

# 生成两类数据
X_class0 = np.random.randn(50, 2) + np.array([2, 2])
X_class1 = np.random.randn(50, 2) + np.array([-2, -2])
X = np.vstack([X_class0, X_class1])
y = np.array([0] * 50 + [1] * 50)

# 训练
knn = KNNClassifier(k=5)
knn.fit(X, y)

# 创建网格用于可视化
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 可视化
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X_class0[:, 0], X_class0[:, 1], c='blue', label='类别 0', edgecolors='k')
plt.scatter(X_class1[:, 0], X_class1[:, 1], c='red', label='类别 1', edgecolors='k')
plt.xlabel('特征 1', fontsize=12)
plt.ylabel('特征 2', fontsize=12)
plt.title('KNN 分类决策边界', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 🎯 实践练习

### 练习 1：实现逻辑回归

**任务**：从零实现逻辑回归分类器。

**要求**：
- 实现 sigmoid 函数
- 实现梯度下降训练
- 实现 predict 方法

---

### 练习 2：模型选择实验

**任务**：比较不同模型在同一数据集上的性能。

**要求**：
- 使用 sklearn 的多个分类器
- 交叉验证评估
- 分析结果

---

## 📝 本章小结

### 核心要点

1. **机器学习分类**：监督学习、无监督学习、强化学习
2. **数据集划分**：训练集、验证集、测试集
3. **过拟合与欠拟合**：原因与解决方案
4. **核心概念**：损失函数、优化器、学习率
5. **模型评估**：准确率、精确率、召回率、F1

### 下一章预告

下一章我们将学习神经网络基础，理解深度学习的核心原理。

---

<div align="center">

[⬅️ 上一章](../chapter02-math-foundations/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter04-neural-networks/README.md)

</div>
