# 第6章：循环神经网络

<div align="center">

[⬅️ 上一章](../chapter05-word-embeddings/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter07-attention/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解 RNN 处理序列数据的核心思想
- ✅ 掌握 RNN 的数学原理和计算过程
- ✅ 理解梯度消失/爆炸问题及解决方案
- ✅ 掌握 LSTM 和 GRU 的工作原理
- ✅ 从零实现 RNN、LSTM 和 GRU
- ✅ 使用 RNN 完成文本生成任务

---

## 🎯 本章内容

### 1. 为什么需要 RNN？

#### 1.1 传统神经网络的局限

在前面的章节中，我们学习了全连接神经网络（MLP）。这种网络有一个重要假设：**输入之间是独立的**。

```
传统 MLP：
输入 x₁, x₂, ..., xₙ → 网络 → 输出 y

每个输入样本独立处理，不考虑样本之间的顺序关系。
```

但在现实世界中，很多数据具有**序列特性**：

| 数据类型 | 特点 | 示例 |
|---------|------|------|
| **文本** | 词与词之间有顺序关系 | "我喜欢你" ≠ "你喜欢我" |
| **语音** | 声音信号是连续的 | 识别"你好"需要理解声音序列 |
| **时间序列** | 时间点之间有关联 | 股票价格、天气预报 |
| **视频** | 帧与帧之间有连续性 | 动作识别 |

```
例子：预测下一个词

输入："今天天气很"
期望输出："好"

要预测"好"，需要理解前面几个词的含义和顺序。
传统 MLP 无法有效处理这种依赖关系。
```

#### 1.2 RNN 的核心思想

**循环神经网络（Recurrent Neural Network, RNN）** 的核心思想是：**引入"记忆"机制，让网络能够保存之前的信息**。

```
传统神经网络：
x → [网络] → y
（输入直接映射到输出）

RNN：
x₁ → [网络] → h₁ → y₁
       ↓
x₂ → [网络] → h₂ → y₂
       ↓
x₃ → [网络] → h₃ → y₃
       ↓
      ...

h 是隐藏状态，保存了之前的信息，传递给下一时刻。
```

**RNN 的三个关键特点**：

1. **时序处理**：按时间步依次处理序列中的每个元素
2. **状态传递**：隐藏状态在时间步之间传递，保存历史信息
3. **参数共享**：所有时间步使用相同的权重，减少参数量

---

### 2. RNN 的结构与数学表达

#### 2.1 基本结构

RNN 的基本单元在每个时间步接收两个输入：
- 当前时刻的输入 **xₜ**
- 上一时刻的隐藏状态 **hₜ₋₁**

```
               hₜ
                ↑
         ┌──────┴──────┐
         │             │
    hₜ₋₁ →[RNN Cell]← xₜ
         │             │
         └─────────────┘
```

#### 2.2 数学公式

RNN 的核心计算公式：

```
隐藏状态更新：
hₜ = tanh(Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + bₕ)

输出计算：
yₜ = Wₕᵧ · hₜ + bᵧ

其中：
- Wₓₕ: 输入到隐藏层的权重
- Wₕₕ: 隐藏层到隐藏层的权重（循环连接）
- Wₕᵧ: 隐藏层到输出的权重
- bₕ, bᵧ: 偏置项
- tanh: 激活函数
```

**参数维度说明**：

```
假设：
- 输入维度: d
- 隐藏状态维度: h
- 输出维度: o

则：
- Wₓₕ: (d, h)
- Wₕₕ: (h, h)
- Wₕᵧ: (h, o)
- bₕ: (h,)
- bᵧ: (o,)

总参数量: d·h + h·h + h·o + h + o
```

#### 2.3 从零实现 RNN

```python
import numpy as np

class SimpleRNN:
    """从零实现简单的 RNN"""
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
            output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重（Xavier 初始化）
        self.W_xh = np.random.randn(input_size, hidden_size) * \
                    np.sqrt(2.0 / (input_size + hidden_size))
        self.W_hh = np.random.randn(hidden_size, hidden_size) * \
                    np.sqrt(2.0 / (hidden_size + hidden_size))
        self.W_hy = np.random.randn(hidden_size, output_size) * \
                    np.sqrt(2.0 / (hidden_size + output_size))
        
        # 初始化偏置
        self.b_h = np.zeros(hidden_size)
        self.b_y = np.zeros(output_size)
        
        # 存储中间结果（用于反向传播）
        self.h_states = None
        self.inputs = None
    
    def tanh(self, x):
        """tanh 激活函数"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """tanh 导数"""
        return 1 - np.tanh(x) ** 2
    
    def forward(self, x, h_prev=None):
        """
        前向传播（单个时间步）
        
        参数:
            x: 当前时刻输入, shape (batch_size, input_size)
            h_prev: 上一时刻隐藏状态, shape (batch_size, hidden_size)
        
        返回:
            y: 当前时刻输出, shape (batch_size, output_size)
            h: 当前时刻隐藏状态, shape (batch_size, hidden_size)
        """
        if h_prev is None:
            h_prev = np.zeros((x.shape[0], self.hidden_size))
        
        # 计算隐藏状态
        # h = tanh(W_xh · x + W_hh · h_prev + b_h)
        z = np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h
        h = self.tanh(z)
        
        # 计算输出
        y = np.dot(h, self.W_hy) + self.b_y
        
        return y, h
    
    def forward_sequence(self, X):
        """
        前向传播（整个序列）
        
        参数:
            X: 输入序列, shape (seq_len, batch_size, input_size)
        
        返回:
            outputs: 所有时间步的输出, shape (seq_len, batch_size, output_size)
            h_states: 所有时间步的隐藏状态, shape (seq_len, batch_size, hidden_size)
        """
        seq_len = X.shape[0]
        batch_size = X.shape[1]
        
        # 存储结果
        outputs = np.zeros((seq_len, batch_size, self.output_size))
        h_states = np.zeros((seq_len + 1, batch_size, self.hidden_size))
        
        # 初始隐藏状态
        h = np.zeros((batch_size, self.hidden_size))
        h_states[0] = h
        
        # 逐时间步计算
        for t in range(seq_len):
            y, h = self.forward(X[t], h)
            outputs[t] = y
            h_states[t + 1] = h
        
        # 保存用于反向传播
        self.h_states = h_states
        self.inputs = X
        
        return outputs, h_states
    
    def backward_sequence(self, d_outputs, learning_rate=0.01):
        """
        反向传播（BPTT: Backpropagation Through Time）
        
        参数:
            d_outputs: 输出的梯度, shape (seq_len, batch_size, output_size)
            learning_rate: 学习率
        """
        seq_len = d_outputs.shape[0]
        batch_size = d_outputs.shape[1]
        
        # 初始化梯度
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # 下一时刻传回的隐藏状态梯度
        dh_next = np.zeros((batch_size, self.hidden_size))
        
        # 反向传播（从最后一个时间步开始）
        for t in reversed(range(seq_len)):
            # 输出层梯度
            dy = d_outputs[t]
            
            # 输出层参数梯度
            dW_hy += np.dot(self.h_states[t + 1].T, dy)
            db_y += np.sum(dy, axis=0)
            
            # 隐藏层梯度
            dh = np.dot(dy, self.W_hy.T) + dh_next
            
            # tanh 导数
            dh_raw = dh * self.tanh_derivative(
                np.dot(self.inputs[t], self.W_xh) + 
                np.dot(self.h_states[t], self.W_hh) + self.b_h
            )
            
            # 隐藏层参数梯度
            dW_xh += np.dot(self.inputs[t].T, dh_raw)
            dW_hh += np.dot(self.h_states[t].T, dh_raw)
            db_h += np.sum(dh_raw, axis=0)
            
            # 传给上一时刻的梯度
            dh_next = np.dot(dh_raw, self.W_hh.T)
        
        # 梯度裁剪（防止梯度爆炸）
        for grad in [dW_xh, dW_hh, dW_hy]:
            np.clip(grad, -5, 5, out=grad)
        
        # 更新参数
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y


# 测试 RNN
np.random.seed(42)

# 创建 RNN
rnn = SimpleRNN(input_size=3, hidden_size=4, output_size=2)

# 创建测试数据（序列长度=5，批量大小=2）
X = np.random.randn(5, 2, 3)

# 前向传播
outputs, h_states = rnn.forward_sequence(X)

print("输入序列形状:", X.shape)
print("输出序列形状:", outputs.shape)
print("隐藏状态形状:", h_states.shape)
print("\n最后一时刻输出:")
print(outputs[-1])
```

---

### 3. RNN 的训练：BPTT

#### 3.1 时间反向传播（BPTT）

RNN 的训练使用 **BPTT（Backpropagation Through Time）**，本质上是将 RNN 按时间展开，然后使用标准的反向传播。

```
RNN 按时间展开：

x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → ...
        ↓           ↓           ↓
       y₁          y₂          y₃

展开后可以看作一个深度网络，每层对应一个时间步。
反向传播时，梯度沿着时间维度反向传递。
```

#### 3.2 梯度消失与梯度爆炸

RNN 的一个核心问题是**梯度消失**和**梯度爆炸**。

```
考虑隐藏状态的梯度传递：

dhₜ/dhₜ₋₁ = Wₕₕ · diag(tanh'(zₜ))

经过 T 个时间步后：

dhₜ/dhₜ₋ₜ = (Wₕₕ · diag(tanh'))^T

问题：
- 如果 ||Wₕₕ|| < 1，梯度呈指数衰减 → 梯度消失
- 如果 ||Wₕₕ|| > 1，梯度呈指数增长 → 梯度爆炸
```

**梯度消失的后果**：
- 无法学习长距离依赖
- RNN 难以记住很久之前的信息

**解决方案**：
1. **梯度裁剪**：限制梯度的最大值，防止爆炸
2. **更好的架构**：LSTM、GRU 解决消失问题

```python
def gradient_clipping(grads, max_norm):
    """梯度裁剪"""
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for grad in grads:
            grad *= scale
    
    return grads

# 示例
grads = [np.random.randn(10, 10) * 10 for _ in range(3)]
print("裁剪前梯度范数:", np.sqrt(sum(np.sum(g**2) for g in grads)))

grads = gradient_clipping(grads, max_norm=5.0)
print("裁剪后梯度范数:", np.sqrt(sum(np.sum(g**2) for g in grads)))
```

---

### 4. 长短时记忆网络（LSTM）

#### 4.1 LSTM 的设计思想

LSTM（Long Short-Term Memory）通过引入**门控机制**来解决梯度消失问题，让网络能够选择性地记住和遗忘信息。

```
LSTM 的核心组件：

1. 遗忘门（Forget Gate）：决定丢弃哪些信息
2. 输入门（Input Gate）：决定存储哪些新信息
3. 细胞状态（Cell State）：长期记忆
4. 输出门（Output Gate）：决定输出哪些信息
```

#### 4.2 LSTM 的结构

```
LSTM 单元结构：

    cₜ₋₁ ──→[×]──→ cₜ ──→
              ↑           ↓
    hₜ₋₁ ──→[LSTM]──→ hₜ ──→
              ↑
             xₜ

门控：
- fₜ (遗忘门): 控制从 cₜ₋₁ 遗忘多少
- iₜ (输入门): 控制写入多少新信息
- oₜ (输出门): 控制输出多少信息
```

#### 4.3 LSTM 的数学公式

```
遗忘门：
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)

输入门：
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)

候选细胞状态：
c̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)

更新细胞状态：
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ

输出门：
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)

隐藏状态：
hₜ = oₜ ⊙ tanh(cₜ)

其中：
- σ: sigmoid 函数
- ⊙: 逐元素乘法
- [hₜ₋₁, xₜ]: 拼接操作
```

#### 4.4 从零实现 LSTM

```python
class LSTM:
    """从零实现 LSTM"""
    
    def __init__(self, input_size, hidden_size):
        """
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 合并所有门的权重（效率更高）
        # W_f, W_i, W_c, W_o 合并为一个大矩阵
        self.W = np.random.randn(input_size + hidden_size, 4 * hidden_size) * \
                 np.sqrt(2.0 / (input_size + hidden_size))
        self.b = np.zeros(4 * hidden_size)
        
        # 门索引
        self.f_idx = slice(0, hidden_size)
        self.i_idx = slice(hidden_size, 2 * hidden_size)
        self.c_idx = slice(2 * hidden_size, 3 * hidden_size)
        self.o_idx = slice(3 * hidden_size, 4 * hidden_size)
    
    def sigmoid(self, x):
        """Sigmoid 函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x, states=None):
        """
        前向传播（单个时间步）
        
        参数:
            x: 输入, shape (batch_size, input_size)
            states: (h_prev, c_prev)
        
        返回:
            h: 隐藏状态, shape (batch_size, hidden_size)
            c: 细胞状态, shape (batch_size, hidden_size)
        """
        if states is None:
            h_prev = np.zeros((x.shape[0], self.hidden_size))
            c_prev = np.zeros((x.shape[0], self.hidden_size))
        else:
            h_prev, c_prev = states
        
        # 拼接输入和上一时刻隐藏状态
        combined = np.concatenate([h_prev, x], axis=1)
        
        # 计算所有门
        gates = np.dot(combined, self.W) + self.b
        
        # 分离各门
        f = self.sigmoid(gates[:, self.f_idx])  # 遗忘门
        i = self.sigmoid(gates[:, self.i_idx])  # 输入门
        c_tilde = np.tanh(gates[:, self.c_idx]) # 候选细胞状态
        o = self.sigmoid(gates[:, self.o_idx])  # 输出门
        
        # 更新细胞状态
        c = f * c_prev + i * c_tilde
        
        # 计算隐藏状态
        h = o * np.tanh(c)
        
        return h, c
    
    def forward_sequence(self, X):
        """
        前向传播（整个序列）
        
        参数:
            X: 输入序列, shape (seq_len, batch_size, input_size)
        
        返回:
            h_states: 隐藏状态序列, shape (seq_len, batch_size, hidden_size)
        """
        seq_len = X.shape[0]
        batch_size = X.shape[1]
        
        h_states = np.zeros((seq_len, batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        h = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_len):
            h, c = self.forward(X[t], (h, c))
            h_states[t] = h
        
        return h_states


# 测试 LSTM
np.random.seed(42)

lstm = LSTM(input_size=3, hidden_size=4)
X = np.random.randn(5, 2, 3)  # (seq_len, batch_size, input_size)

h_states = lstm.forward_sequence(X)
print("LSTM 隐藏状态形状:", h_states.shape)
print("\n最后时刻隐藏状态:")
print(h_states[-1])
```

#### 4.5 为什么 LSTM 能解决梯度消失？

```
关键在于细胞状态的更新：

cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ

梯度沿细胞状态传递：
∂cₜ/∂cₜ₋₁ = fₜ

当 fₜ ≈ 1 时，梯度可以无损传递（"恒等映射"）
这允许网络学习长距离依赖。
```

---

### 5. 门控循环单元（GRU）

#### 5.1 GRU 的简化设计

GRU（Gated Recurrent Unit）是 LSTM 的简化版本，参数更少，计算更快。

```
GRU 只有两个门：
- 重置门（Reset Gate）：控制是否忽略之前的隐藏状态
- 更新门（Update Gate）：控制新旧状态的混合比例
```

#### 5.2 GRU 的数学公式

```
重置门：
rₜ = σ(Wr · [hₜ₋₁, xₜ])

更新门：
zₜ = σ(Wz · [hₜ₋₁, xₜ])

候选隐藏状态：
h̃ₜ = tanh(Wh · [rₜ ⊙ hₜ₋₁, xₜ])

最终隐藏状态：
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ

GRU 比 LSTM 少了一个门，参数量减少约 25%。
```

#### 5.3 从零实现 GRU

```python
class GRU:
    """从零实现 GRU"""
    
    def __init__(self, input_size, hidden_size):
        """
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 重置门和更新门权重
        self.W_rz = np.random.randn(input_size + hidden_size, 2 * hidden_size) * \
                    np.sqrt(2.0 / (input_size + hidden_size))
        self.b_rz = np.zeros(2 * hidden_size)
        
        # 候选隐藏状态权重
        self.W_h = np.random.randn(input_size + hidden_size, hidden_size) * \
                   np.sqrt(2.0 / (input_size + hidden_size))
        self.b_h = np.zeros(hidden_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x, h_prev=None):
        """
        前向传播（单个时间步）
        
        参数:
            x: 输入, shape (batch_size, input_size)
            h_prev: 上一时刻隐藏状态
        
        返回:
            h: 当前隐藏状态
        """
        if h_prev is None:
            h_prev = np.zeros((x.shape[0], self.hidden_size))
        
        # 拼接
        combined = np.concatenate([h_prev, x], axis=1)
        
        # 重置门和更新门
        rz = np.dot(combined, self.W_rz) + self.b_rz
        r = self.sigmoid(rz[:, :self.hidden_size])  # 重置门
        z = self.sigmoid(rz[:, self.hidden_size:])  # 更新门
        
        # 候选隐藏状态
        combined_r = np.concatenate([r * h_prev, x], axis=1)
        h_tilde = np.tanh(np.dot(combined_r, self.W_h) + self.b_h)
        
        # 最终隐藏状态
        h = (1 - z) * h_prev + z * h_tilde
        
        return h
    
    def forward_sequence(self, X):
        """前向传播（整个序列）"""
        seq_len = X.shape[0]
        batch_size = X.shape[1]
        
        h_states = np.zeros((seq_len, batch_size, self.hidden_size))
        h = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_len):
            h = self.forward(X[t], h)
            h_states[t] = h
        
        return h_states


# 测试 GRU
np.random.seed(42)

gru = GRU(input_size=3, hidden_size=4)
X = np.random.randn(5, 2, 3)

h_states = gru.forward_sequence(X)
print("GRU 隐藏状态形状:", h_states.shape)
print("\n最后时刻隐藏状态:")
print(h_states[-1])
```

---

### 6. 双向 RNN 与深层 RNN

#### 6.1 双向 RNN

在某些任务中，当前时刻的输出可能依赖于未来的信息。

```
填空任务：
"我喜欢吃__和苹果"

要预测空白处，需要知道后面的"苹果"。
单向 RNN 只能看到前面的"我喜欢吃"。
双向 RNN 同时看到前后文。
```

```python
class BidirectionalRNN:
    """双向 RNN"""
    
    def __init__(self, input_size, hidden_size):
        self.forward_rnn = SimpleRNN(input_size, hidden_size, hidden_size)
        self.backward_rnn = SimpleRNN(input_size, hidden_size, hidden_size)
        self.hidden_size = hidden_size
    
    def forward_sequence(self, X):
        """前向传播"""
        # 正向
        _, h_forward = self.forward_rnn.forward_sequence(X)
        h_forward = h_forward[1:]
        
        # 逆向
        X_reversed = X[::-1]
        _, h_backward = self.backward_rnn.forward_sequence(X_reversed)
        h_backward = h_backward[1:]
        h_backward = h_backward[::-1]
        
        # 拼接
        h_states = np.concatenate([h_forward, h_backward], axis=2)
        return h_states
```

#### 6.2 深层 RNN

堆叠多层 RNN 形成深层网络。

```python
class DeepRNN:
    """深层 RNN"""
    
    def __init__(self, input_size, hidden_size, num_layers):
        self.layers = []
        self.layers.append(SimpleRNN(input_size, hidden_size, hidden_size))
        
        for _ in range(num_layers - 1):
            self.layers.append(SimpleRNN(hidden_size, hidden_size, hidden_size))
    
    def forward_sequence(self, X):
        h = X
        for layer in self.layers:
            _, h_states = layer.forward_sequence(h)
            h = h_states[1:]
        return h
```

---

## 💻 完整代码示例

### 示例：字符级文本生成

```python
"""
完整示例：使用字符级 RNN 生成文本
"""
import numpy as np

class CharRNN:
    """字符级 RNN 文本生成模型"""
    
    def __init__(self, vocab_size, hidden_size, seq_length):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        # 初始化参数
        self.Wxh = np.random.randn(vocab_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, vocab_size) * 0.01
        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(vocab_size)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    
    def forward(self, inputs, h_prev):
        """前向传播"""
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        
        for t in range(len(inputs)):
            xs[t] = np.zeros(self.vocab_size)
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(xs[t], self.Wxh) + 
                           np.dot(hs[t-1], self.Whh) + self.bh)
            ys[t] = np.dot(hs[t], self.Why) + self.by
            ps[t] = self.softmax(ys[t])
        
        return xs, hs, ps
    
    def loss(self, ps, targets):
        loss = 0
        for t in range(len(targets)):
            loss += -np.log(ps[t][targets[t]] + 1e-10)
        return loss / len(targets)
    
    def train(self, data, char_to_ix, ix_to_char, epochs=100, 
              learning_rate=0.1, print_every=10):
        """训练模型"""
        h_prev = np.zeros(self.hidden_size)
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length
        
        # Adagrad 内存
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)
        
        for epoch in range(epochs):
            for p in range(0, len(data) - self.seq_length, self.seq_length):
                inputs = [char_to_ix[ch] for ch in data[p:p+self.seq_length]]
                targets = [char_to_ix[ch] for ch in data[p+1:p+self.seq_length+1]]
                
                xs, hs, ps = self.forward(inputs, h_prev)
                loss = self.loss(ps, targets)
                smooth_loss = smooth_loss * 0.999 + loss * 0.001
                
                # 反向传播（简化版）
                dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
                dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
                dh_next = np.zeros_like(hs[0])
                
                for t in reversed(range(len(targets))):
                    dy = np.copy(ps[t])
                    dy[targets[t]] -= 1
                    dWhy += np.outer(hs[t], dy)
                    dby += dy
                    dh = np.dot(dy, self.Why.T) + dh_next
                    dh_raw = (1 - hs[t] * hs[t]) * dh
                    dWxh += np.outer(xs[t], dh_raw)
                    dWhh += np.outer(hs[t-1], dh_raw)
                    dbh += dh_raw
                    dh_next = np.dot(dh_raw, self.Whh.T)
                
                # 梯度裁剪
                for grad in [dWxh, dWhh, dWhy, dbh, dby]:
                    np.clip(grad, -5, 5, out=grad)
                
                # Adagrad 更新
                for param, dparam, mem in zip(
                    [self.Wxh, self.Whh, self.Why, self.bh, self.by],
                    [dWxh, dWhh, dWhy, dbh, dby],
                    [mWxh, mWhh, mWhy, mbh, mby]
                ):
                    mem += dparam * dparam
                    param -= learning_rate * dparam / np.sqrt(mem + 1e-8)
                
                h_prev = hs[len(inputs) - 1]
            
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}, Loss: {smooth_loss:.4f}")
                sample = self.sample(h_prev, inputs[0], 50, ix_to_char)
                print(f"Generated: {sample}\n")
    
    def sample(self, h, seed_ix, n, ix_to_char):
        """生成文本"""
        x = np.zeros(self.vocab_size)
        x[seed_ix] = 1
        ixes = []
        
        for _ in range(n):
            h = np.tanh(np.dot(x, self.Wxh) + np.dot(h, self.Whh) + self.bh)
            y = np.dot(h, self.Why) + self.by
            p = self.softmax(y)
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros(self.vocab_size)
            x[ix] = 1
            ixes.append(ix)
        
        return ''.join(ix_to_char[ix] for ix in ixes)


# 测试
text = """
hello world
hello python
hello machine learning
hello deep learning
""" * 5

chars = list(set(text))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"词汇表大小: {vocab_size}")

model = CharRNN(vocab_size, hidden_size=64, seq_length=20)
model.train(text, char_to_ix, ix_to_char, epochs=50, print_every=10)

print("=== 生成文本 ===")
h = np.zeros(64)
generated = model.sample(h, char_to_ix['h'], 50, ix_to_char)
print(f"Generated: {generated}")
```

---

## 🎯 实践练习

### 练习 1：情感分析

**任务**：使用 RNN 对电影评论进行情感分类。

### 练习 2：模型比较

**任务**：在相同任务上比较 RNN、LSTM、GRU 的性能。

### 练习 3：序列到序列模型

**任务**：实现简单的 Encoder-Decoder 结构用于机器翻译。

---

## 📝 本章小结

### 核心要点

1. **RNN 核心思想**：通过隐藏状态传递历史信息，处理序列数据
2. **BPTT**：时间反向传播，按时间展开后使用标准反向传播
3. **梯度问题**：长期依赖导致梯度消失/爆炸
4. **LSTM**：通过门控机制（遗忘门、输入门、输出门）解决梯度消失
5. **GRU**：LSTM 的简化版本，只有重置门和更新门

### 关键公式

```
RNN:
hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + bₕ)

LSTM:
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
cₜ = fₜ⊙cₜ₋₁ + iₜ⊙tanh(Wc·[hₜ₋₁, xₜ] + bc)
hₜ = oₜ⊙tanh(cₜ)

GRU:
hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙tanh(Wh·[rₜ⊙hₜ₋₁, xₜ])
```

### 模型对比

| 特性 | RNN | LSTM | GRU |
|------|-----|------|-----|
| 参数量 | 少 | 多 | 中 |
| 训练速度 | 快 | 慢 | 中 |
| 长期依赖 | 差 | 好 | 好 |
| 门控数量 | 0 | 3 | 2 |

---

<div align="center">

[⬅️ 上一章](../chapter05-word-embeddings/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter07-attention/README.md)

</div>
