# 第4章：神经网络基础

<div align="center">

[⬅️ 上一章](../chapter03-ml-fundamentals/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter05-word-embeddings/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解感知机的工作原理
- ✅ 掌握多层神经网络的结构
- ✅ 理解前向传播和反向传播
- ✅ 掌握常用激活函数
- ✅ 从零实现一个神经网络

---

## 🎯 本章内容

### 1. 感知机（Perceptron）

#### 1.1 感知机原理

感知机是最简单的神经网络，只有一个神经元。

```
输入: x₁, x₂, ..., xₙ
权重: w₁, w₂, ..., wₙ
偏置: b
激活函数: f

输出: y = f(Σwᵢxᵢ + b)
```

```python
import numpy as np

class Perceptron:
    """感知机"""
    
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size)
        self.bias = 0
        self.lr = learning_rate
    
    def activation(self, x):
        """阶跃激活函数"""
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        """预测"""
        z = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(z_i) for z_i in z])
    
    def fit(self, X, y, epochs=100):
        """训练"""
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                # 预测
                y_pred = self.predict([xi])[0]
                
                # 更新权重（感知机学习规则）
                error = yi - y_pred
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

# 测试：AND 逻辑门
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size=2)
perceptron.fit(X, y, epochs=10)

print("AND 逻辑门:")
for xi, yi in zip(X, y):
    pred = perceptron.predict([xi])[0]
    print(f"  {xi} -> 预测: {pred}, 真实: {yi}")
```

**感知机的局限性**：

```
感知机无法解决 XOR 问题：
XOR: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0

这是线性不可分的！
需要多层神经网络。
```

---

### 2. 多层神经网络

#### 2.1 多层感知机（MLP）

```python
import numpy as np

class NeuralNetwork:
    """多层神经网络"""
    
    def __init__(self, layer_sizes):
        """
        参数:
            layer_sizes: 每层神经元数量
                        例如 [2, 4, 1] 表示输入层2个，隐藏层4个，输出层1个
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # 初始化权重和偏置
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        """Sigmoid 激活函数"""
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        """Sigmoid 导数"""
        return a * (1 - a)
    
    def forward(self, X):
        """前向传播"""
        self.activations = [X]
        self.z_values = []
        
        a = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            self.z_values.append(z)
            a = self.sigmoid(z)
            self.activations.append(a)
        
        return a
    
    def backward(self, X, y, learning_rate=0.1):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层误差
        output = self.activations[-1]
        delta = (output - y) * self.sigmoid_derivative(output)
        
        # 存储梯度
        gradients = []
        
        # 反向计算梯度
        for i in range(len(self.weights) - 1, -1, -1):
            # 计算梯度
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients.insert(0, (dw, db))
            
            # 传播到上一层
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * \
                        self.sigmoid_derivative(self.activations[i])
        
        # 更新参数
        for i, (dw, db) in enumerate(gradients):
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
    
    def train(self, X, y, epochs=1000, learning_rate=0.1):
        """训练"""
        losses = []
        
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            
            # 计算损失
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            
            # 反向传播
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """预测"""
        return self.forward(X)

# 测试：XOR 问题
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 4, 1])
losses = nn.train(X, y, epochs=5000, learning_rate=1.0)

# 预测
print("\nXOR 预测结果:")
for xi, yi in zip(X, y):
    pred = nn.predict(xi.reshape(1, -1))[0][0]
    print(f"  {xi} -> 预测: {pred:.4f}, 真实: {yi[0]}")
```

---

### 3. 激活函数

#### 3.1 常用激活函数

```python
import numpy as np
import matplotlib.pyplot as plt

# 激活函数定义
def sigmoid(x):
    """Sigmoid: 值域 (0, 1)"""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh: 值域 (-1, 1)"""
    return np.tanh(x)

def relu(x):
    """ReLU: 值域 [0, +∞)"""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU"""
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    """Softmax: 多分类"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# 可视化
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
    plt.plot(x, y, linewidth=2)
    plt.title(f'{name} 激活函数', fontsize=12)
    plt.xlabel('x', fontsize=10)
    plt.ylabel('f(x)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()
```

**激活函数选择**：

| 激活函数 | 优点 | 缺点 | 适用场景 |
|---------|------|------|----------|
| **Sigmoid** | 输出在(0,1)，适合概率 | 梯度消失，输出非零中心 | 二分类输出层 |
| **Tanh** | 零中心，收敛快 | 梯度消失 | RNN |
| **ReLU** | 计算快，缓解梯度消失 | 神经元死亡 | 深度网络隐藏层 |
| **Leaky ReLU** | 解决神经元死亡 | 需调参 | 深度网络 |
| **Softmax** | 输出和为1 | - | 多分类输出层 |

---

### 4. 前向传播与反向传播

#### 4.1 前向传播

```
输入: X
隐藏层: Z₁ = X·W₁ + b₁, A₁ = f(Z₁)
输出层: Z₂ = A₁·W₂ + b₂, A₂ = f(Z₂)

参数更新需要反向传播计算梯度。
```

#### 4.2 反向传播

```
链式法则:
∂L/∂W = ∂L/∂A · ∂A/∂Z · ∂Z/∂W

梯度从输出层向输入层反向传播。
```

```python
def backpropagation_demo():
    """反向传播演示"""
    
    # 简单的 2 层网络
    # y = sigmoid(sigmoid(x·W₁)·W₂)
    
    x = np.array([[0.5]])
    y_true = np.array([[1.0]])
    
    # 初始化权重
    W1 = np.array([[0.8]])
    W2 = np.array([[1.2]])
    
    # 学习率
    lr = 0.5
    
    print("初始权重:")
    print(f"W1 = {W1[0,0]:.4f}, W2 = {W2[0,0]:.4f}")
    
    for epoch in range(5):
        # ===== 前向传播 =====
        z1 = np.dot(x, W1)         # 第一层线性变换
        a1 = sigmoid(z1)           # 第一层激活
        z2 = np.dot(a1, W2)        # 第二层线性变换
        a2 = sigmoid(z2)           # 第二层激活（输出）
        
        # 计算损失
        loss = 0.5 * (y_true - a2) ** 2
        
        # ===== 反向传播 =====
        # 输出层误差
        delta2 = (a2 - y_true) * a2 * (1 - a2)  # dL/dz2
        
        # 隐藏层误差
        delta1 = np.dot(delta2, W2.T) * a1 * (1 - a1)  # dL/dz1
        
        # 计算梯度
        dW2 = np.dot(a1.T, delta2)
        dW1 = np.dot(x.T, delta1)
        
        # 更新权重
        W2 -= lr * dW2
        W1 -= lr * dW1
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  预测: {a2[0,0]:.4f}, 损失: {loss[0,0]:.6f}")
        print(f"  W1 = {W1[0,0]:.4f}, W2 = {W2[0,0]:.4f}")

backpropagation_demo()
```

---

### 5. 深度学习技巧

#### 5.1 Batch Normalization

```python
class BatchNormalization:
    """批量归一化"""
    
    def __init__(self, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = None
        self.running_var = None
        self.gamma = None
        self.beta = None
    
    def initialize(self, num_features):
        """初始化参数"""
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x, training=True):
        """前向传播"""
        if training:
            # 训练模式：使用当前批次的统计量
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            # 更新运行统计量
            self.running_mean = self.momentum * self.running_mean + \
                               (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + \
                              (1 - self.momentum) * var
        else:
            # 推理模式：使用运行统计量
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        
        # 缩放和平移
        out = self.gamma * x_norm + self.beta
        
        return out
```

#### 5.2 Dropout

```python
class Dropout:
    """Dropout 正则化"""
    
    def __init__(self, drop_prob=0.5):
        self.drop_prob = drop_prob
        self.mask = None
    
    def forward(self, x, training=True):
        """前向传播"""
        if training:
            # 训练模式：随机丢弃神经元
            self.mask = (np.random.rand(*x.shape) > self.drop_prob) / \
                       (1 - self.drop_prob)
            return x * self.mask
        else:
            # 推理模式：不丢弃
            return x

# 示例
x = np.random.randn(10, 5)
dropout = Dropout(drop_prob=0.3)

print("训练模式:")
print(dropout.forward(x, training=True))

print("\n推理模式:")
print(dropout.forward(x, training=False))
```

---

## 💻 完整代码示例

### 示例：手写数字识别

```python
"""
完整示例：使用神经网络识别手写数字（简化版）
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SimpleNN:
    """简单的全连接神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size):
        """初始化"""
        # 权重初始化（Xavier）
        self.W1 = np.random.randn(input_size, hidden_size) * \
                  np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * \
                  np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.loss_history = []
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """前向传播"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate=0.01):
        """反向传播"""
        m = X.shape[0]
        
        # One-hot 编码
        y_onehot = np.zeros_like(self.a2)
        y_onehot[np.arange(m), y] = 1
        
        # 输出层梯度
        dz2 = self.a2 - y_onehot
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 隐藏层梯度
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 更新参数
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=100, learning_rate=0.1, batch_size=32):
        """训练"""
        m = X.shape[0]
        
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # 批量训练
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 前向传播
                output = self.forward(X_batch)
                
                # 反向传播
                self.backward(X_batch, y_batch, learning_rate)
            
            # 计算损失
            if epoch % 10 == 0:
                output = self.forward(X)
                y_onehot = np.zeros_like(output)
                y_onehot[np.arange(m), y] = 1
                loss = -np.mean(y_onehot * np.log(output + 1e-10))
                acc = np.mean(np.argmax(output, axis=1) == y)
                
                self.loss_history.append(loss)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    def predict(self, X):
        """预测"""
        output = self.forward(X)
        return np.argmax(output, axis=1)

# 加载数据
digits = load_digits()
X = digits.data / 16.0  # 归一化
y = digits.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# 创建并训练模型
nn = SimpleNN(input_size=64, hidden_size=128, output_size=10)
nn.train(X_train, y_train, epochs=100, learning_rate=0.1, batch_size=32)

# 测试
predictions = nn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"\n测试集准确率: {accuracy:.4f}")

# 可视化一些预测结果
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"预测: {predictions[i]}, 真实: {y_test[i]}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('digit_recognition.png', dpi=300)
print("\n预测结果已保存为 digit_recognition.png")
plt.show()
```

---

## 🎯 实践练习

### 练习 1：实现不同激活函数

**任务**：实现并比较不同激活函数的效果。

### 练习 2：调整网络结构

**任务**：修改隐藏层大小和数量，观察性能变化。

---

## 📝 本章小结

### 核心要点

1. **感知机**：最简单的神经网络，只能解决线性可分问题
2. **多层网络**：通过隐藏层解决非线性问题
3. **激活函数**：引入非线性，ReLU 是主流选择
4. **前向传播**：输入到输出的计算过程
5. **反向传播**：通过链式法则计算梯度

### 关键公式

```
前向传播: a = f(W·x + b)
反向传播: ∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
权重更新: W_new = W_old - α·∂L/∂W
```

---

<div align="center">

[⬅️ 上一章](../chapter03-ml-fundamentals/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter05-word-embeddings/README.md)

</div>
