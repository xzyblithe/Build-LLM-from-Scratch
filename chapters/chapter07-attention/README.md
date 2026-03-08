# 第7章：Attention 机制

<div align="center">

[⬅️ 上一章](../chapter06-rnn/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter08-transformer/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解 Attention 机制的动机与核心思想
- ✅ 掌握注意力分数的计算方法
- ✅ 深入理解 Self-Attention（自注意力）
- ✅ 掌握 Multi-Head Attention（多头注意力）
- ✅ 了解各种注意力变体（Additive、Scaled Dot-Product 等）
- ✅ 从零实现完整的注意力机制
- ✅ 理解 Attention 在 NLP 和 CV 中的应用

---

## 🎯 本章内容

### 1. 为什么需要 Attention？

#### 1.1 Seq2Seq 的瓶颈

在传统的序列到序列（Seq2Seq）模型中，编码器将整个输入序列压缩成一个固定长度的向量，解码器从这个向量生成输出。

```
传统 Seq2Seq：

编码器: x₁, x₂, ..., xₙ → [固定长度向量 c]
解码器: c → y₁, y₂, ..., yₘ

问题：
1. 信息瓶颈：长序列信息无法完整保存在固定向量中
2. 长距离依赖：解码时"忘记"输入的早期信息
```

#### 1.2 Attention 的直觉

**Attention 让模型在生成每个输出时，都能"关注"输入序列的不同部分。**

```
类比：阅读理解

问题："北京是中国的首都，它有多少人口？"

回答时，你会：
- 读到"它"→ 关注"北京"
- 读到"人口"→ 关注数字信息

Attention 让模型学会这种"关注"能力。
```

```
Seq2Seq + Attention：

编码器: x₁, x₂, ..., xₙ → h₁, h₂, ..., hₙ

解码器生成 yₜ 时：
1. 计算当前状态 sₜ 与所有 hᵢ 的相关性（注意力分数）
2. 加权求和得到上下文向量 cₜ
3. 基于 cₜ 和 sₜ 生成 yₜ
```

---

### 2. Attention 的基本形式

#### 2.1 通用框架

Attention 机制可以抽象为：**Query、Key、Value 的交互**

```
给定：
- Query (查询): Q = [q₁, q₂, ..., qₘ]
- Key (键): K = [k₁, k₂, ..., kₙ]
- Value (值): V = [v₁, v₂, ..., vₙ]

计算：
1. 注意力分数: eᵢⱼ = score(qᵢ, kⱼ)
2. 注意力权重: αᵢⱼ = softmax(eᵢⱼ)
3. 加权求和: outputᵢ = Σⱼ αᵢⱼ · vⱼ

直觉：
- Query: "我想找什么"
- Key: "这里有什么"
- Value: "具体内容"
- 匹配 Query 和 Key，提取相关 Value
```

#### 2.2 注意力分数计算方式

**1. 点积注意力（Dot-Product）**

```
score(q, k) = q · k

优点：计算简单，效率高
缺点：向量维度大时分数可能过大
```

**2. 缩放点积注意力（Scaled Dot-Product）**

```
score(q, k) = (q · k) / √dₖ

其中 dₖ 是 Key 的维度

优点：解决了点积分数过大的问题
应用：Transformer 使用的标准方法
```

**3. 加性注意力（Additive / Bahdanau Attention）**

```
score(q, k) = vᵀ · tanh(Wq · q + Wk · k)

优点：可以学习非线性关系
缺点：计算量较大
应用：早期 Seq2Seq 模型
```

---

### 3. Bahdanau Attention

#### 3.1 结构

Bahdanau Attention（2015）是最早的注意力机制之一，用于机器翻译。

```
编码器：双向 RNN
解码器：单向 RNN + Attention

计算过程：
1. 编码器生成隐藏状态 h₁, h₂, ..., hₙ
2. 解码器在时刻 t：
   - 当前状态 sₜ
   - 计算注意力分数: eₜᵢ = vᵀ · tanh(Ws · sₜ + Wh · hᵢ)
   - 注意力权重: αₜᵢ = softmax(eₜᵢ)
   - 上下文向量: cₜ = Σᵢ αₜᵢ · hᵢ
   - 生成输出: yₜ = f(sₜ, cₜ)
```

#### 3.2 实现

```python
import numpy as np

class BahdanauAttention:
    """Bahdanau (Additive) Attention"""
    
    def __init__(self, hidden_size):
        """
        参数:
            hidden_size: 隐藏状态维度
        """
        self.hidden_size = hidden_size
        
        # 参数初始化
        self.W_s = np.random.randn(hidden_size, hidden_size) * 0.1
        self.W_h = np.random.randn(hidden_size, hidden_size) * 0.1
        self.v = np.random.randn(hidden_size) * 0.1
    
    def compute_attention(self, s_t, h_states):
        """
        计算注意力
        
        参数:
            s_t: 解码器当前状态, shape (hidden_size,)
            h_states: 编码器所有隐藏状态, shape (seq_len, hidden_size)
        
        返回:
            context: 上下文向量, shape (hidden_size,)
            weights: 注意力权重, shape (seq_len,)
        """
        seq_len = h_states.shape[0]
        
        # 计算注意力分数
        # score = v^T · tanh(Ws · s_t + Wh · h_i)
        scores = np.zeros(seq_len)
        for i in range(seq_len):
            combined = np.dot(self.W_s, s_t) + np.dot(self.W_h, h_states[i])
            scores[i] = np.dot(self.v, np.tanh(combined))
        
        # Softmax 得到权重
        weights = self._softmax(scores)
        
        # 加权求和得到上下文向量
        context = np.zeros(self.hidden_size)
        for i in range(seq_len):
            context += weights[i] * h_states[i]
        
        return context, weights
    
    def _softmax(self, x):
        """Softmax 函数"""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)


# 测试
np.random.seed(42)

hidden_size = 8
attention = BahdanauAttention(hidden_size)

# 模拟编码器输出
h_states = np.random.randn(5, hidden_size)  # 序列长度 5

# 模拟解码器当前状态
s_t = np.random.randn(hidden_size)

# 计算注意力
context, weights = attention.compute_attention(s_t, h_states)

print("注意力权重:", weights)
print("权重和:", np.sum(weights))
print("上下文向量形状:", context.shape)
```

---

### 4. Luong Attention

#### 4.1 与 Bahdanau 的区别

Luong Attention（2015）提出了几种不同的注意力计算方式。

```
Bahdanau Attention:
- 使用解码器前一时刻状态 s_{t-1} 计算 attention
- 加性注意力

Luong Attention:
- 使用解码器当前时刻状态 s_t 计算 attention
- 支持多种分数计算方式
- 提出了 local attention（局部注意力）
```

#### 4.2 三种分数计算方式

```python
class LuongAttention:
    """Luong Attention"""
    
    def __init__(self, hidden_size, method='dot'):
        """
        参数:
            hidden_size: 隐藏状态维度
            method: 分数计算方法 ('dot', 'general', 'concat')
        """
        self.hidden_size = hidden_size
        self.method = method
        
        if method == 'general':
            self.W = np.random.randn(hidden_size, hidden_size) * 0.1
        elif method == 'concat':
            self.W = np.random.randn(2 * hidden_size, hidden_size) * 0.1
            self.v = np.random.randn(hidden_size) * 0.1
    
    def score(self, s_t, h_i):
        """计算单个注意力分数"""
        if self.method == 'dot':
            return np.dot(s_t, h_i)
        elif self.method == 'general':
            return np.dot(s_t, np.dot(self.W, h_i))
        elif self.method == 'concat':
            combined = np.concatenate([s_t, h_i])
            return np.dot(self.v, np.tanh(np.dot(self.W, combined)))
    
    def compute_attention(self, s_t, h_states):
        """计算注意力"""
        seq_len = h_states.shape[0]
        
        # 计算所有分数
        scores = np.array([self.score(s_t, h_states[i]) for i in range(seq_len)])
        
        # Softmax
        weights = self._softmax(scores)
        
        # 加权求和
        context = np.dot(weights, h_states)
        
        return context, weights
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)


# 测试三种方法
np.random.seed(42)
hidden_size = 8
h_states = np.random.randn(5, hidden_size)
s_t = np.random.randn(hidden_size)

for method in ['dot', 'general', 'concat']:
    attention = LuongAttention(hidden_size, method)
    context, weights = attention.compute_attention(s_t, h_states)
    print(f"\n{method} attention 权重: {weights}")
```

---

### 5. Self-Attention（自注意力）

#### 5.1 核心思想

Self-Attention 让序列中的每个元素都能关注序列中的其他元素。

```
传统 Attention:
- Query 来自解码器
- Key/Value 来自编码器

Self-Attention:
- Query、Key、Value 都来自同一个序列
- 序列中的每个位置都与其他所有位置交互
```

```
示例：句子 "我 喜欢 AI"

Self-Attention 让：
- "我" 关注 "我"、"喜欢"、"AI"
- "喜欢" 关注 "我"、"喜欢"、"AI"
- "AI" 关注 "我"、"喜欢"、"AI"

每个词都能获取上下文信息。
```

#### 5.2 数学表达

```
输入序列: X = [x₁, x₂, ..., xₙ]

线性变换得到 Q, K, V:
Q = X · W_Q
K = X · W_K
V = X · W_V

注意力计算:
Attention(Q, K, V) = softmax(Q · K^T / √dₖ) · V

其中 dₖ 是 Key 的维度
```

#### 5.3 实现

```python
class SelfAttention:
    """Self-Attention 实现"""
    
    def __init__(self, input_size, hidden_size):
        """
        参数:
            input_size: 输入维度
            hidden_size: 隐藏维度（Q, K, V 的维度）
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Q, K, V 的投影矩阵
        self.W_q = np.random.randn(input_size, hidden_size) * 0.1
        self.W_k = np.random.randn(input_size, hidden_size) * 0.1
        self.W_v = np.random.randn(input_size, hidden_size) * 0.1
        
        # 输出投影
        self.W_o = np.random.randn(hidden_size, hidden_size) * 0.1
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入序列, shape (seq_len, input_size)
        
        返回:
            output: 输出序列, shape (seq_len, hidden_size)
            weights: 注意力权重, shape (seq_len, seq_len)
        """
        seq_len = X.shape[0]
        
        # 计算 Q, K, V
        Q = np.dot(X, self.W_q)  # (seq_len, hidden_size)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # 计算注意力分数
        # scores = Q · K^T / √dₖ
        d_k = self.hidden_size
        scores = np.dot(Q, K.T) / np.sqrt(d_k)  # (seq_len, seq_len)
        
        # Softmax
        weights = self._softmax(scores)  # (seq_len, seq_len)
        
        # 加权求和
        output = np.dot(weights, V)  # (seq_len, hidden_size)
        
        # 输出投影
        output = np.dot(output, self.W_o)
        
        return output, weights
    
    def _softmax(self, x):
        """对每行做 softmax"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


# 测试 Self-Attention
np.random.seed(42)

# 输入：3 个词，每个词 4 维向量
X = np.random.randn(3, 4)

self_attn = SelfAttention(input_size=4, hidden_size=8)
output, weights = self_attn.forward(X)

print("输入形状:", X.shape)
print("输出形状:", output.shape)
print("\n注意力权重矩阵:")
print(weights)
print("\n每行权重和:", weights.sum(axis=1))
```

---

### 6. Multi-Head Attention（多头注意力）

#### 6.1 动机

单个 Self-Attention 可能只关注某一方面。Multi-Head 让模型同时从多个角度关注序列。

```
类比：多个视角

看一幅画：
- 头1：关注颜色
- 头2：关注形状
- 头3：关注纹理

多头注意力让模型学习多个不同的"注意力模式"。
```

#### 6.2 结构

```
Multi-Head Attention:

输入 X
  ↓
并行多个头:
Head 1: Q₁=X·W₁^Q, K₁=X·W₁^K, V₁=X·W₁^V → Attention₁
Head 2: Q₂=X·W₂^Q, K₂=X·W₂^K, V₂=X·W₂^V → Attention₂
...
Head h: Q_h=X·W_h^Q, K_h=X·W_h^K, V_h=X·W_h^V → Attention_h
  ↓
拼接: Concat(Attention₁, ..., Attention_h)
  ↓
线性投影: Output = Concat · W_O
```

#### 6.3 实现

```python
class MultiHeadAttention:
    """Multi-Head Attention 实现"""
    
    def __init__(self, input_size, hidden_size, num_heads):
        """
        参数:
            input_size: 输入维度
            hidden_size: 总隐藏维度
            num_heads: 头的数量
        """
        assert hidden_size % num_heads == 0, "hidden_size 必须能被 num_heads 整除"
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 所有头的 Q, K, V 投影（合并为一个矩阵）
        self.W_q = np.random.randn(input_size, hidden_size) * 0.1
        self.W_k = np.random.randn(input_size, hidden_size) * 0.1
        self.W_v = np.random.randn(input_size, hidden_size) * 0.1
        
        # 输出投影
        self.W_o = np.random.randn(hidden_size, hidden_size) * 0.1
    
    def forward(self, X, mask=None):
        """
        前向传播
        
        参数:
            X: 输入, shape (seq_len, input_size)
            mask: 可选的掩码
        """
        seq_len = X.shape[0]
        
        # 计算 Q, K, V
        Q = np.dot(X, self.W_q)  # (seq_len, hidden_size)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # 重塑为多头形式
        # (seq_len, hidden_size) -> (num_heads, seq_len, head_dim)
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        
        # 计算每个头的注意力
        head_outputs = []
        head_weights = []
        
        for h in range(self.num_heads):
            # 计算注意力分数
            scores = np.dot(Q[h], K[h].T) / np.sqrt(self.head_dim)
            
            # 应用掩码（如果有）
            if mask is not None:
                scores = scores + mask * -1e9
            
            # Softmax
            weights = self._softmax(scores)
            
            # 加权求和
            output = np.dot(weights, V[h])
            
            head_outputs.append(output)
            head_weights.append(weights)
        
        # 拼接所有头
        concat = np.concatenate(head_outputs, axis=1)  # (seq_len, hidden_size)
        
        # 输出投影
        output = np.dot(concat, self.W_o)
        
        return output, head_weights
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


# 测试 Multi-Head Attention
np.random.seed(42)

X = np.random.randn(4, 6)  # 序列长度 4，维度 6

mha = MultiHeadAttention(input_size=6, hidden_size=12, num_heads=3)
output, weights = mha.forward(X)

print("输入形状:", X.shape)
print("输出形状:", output.shape)
print("\n每个头的注意力权重:")
for i, w in enumerate(weights):
    print(f"\nHead {i+1}:")
    print(w.round(3))
```

---

### 7. Masked Attention

#### 7.1 为什么需要掩码？

在某些场景下，需要限制注意力的范围：

```
1. Padding Mask:
   - 输入序列可能长度不一，需要 padding
   - padding 位置不应该被关注

2. Causal Mask (Look-ahead Mask):
   - 解码时，当前位置不能看到后面的信息
   - 防止"作弊"
```

#### 7.2 实现

```python
def create_padding_mask(seq, pad_token=0):
    """创建 padding 掩码"""
    # padding 位置为 1，其他位置为 0
    mask = (seq == pad_token).astype(float)
    return mask[:, np.newaxis, :]  # (batch, 1, seq_len)


def create_causal_mask(seq_len):
    """创建因果掩码（上三角矩阵）"""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask


class MaskedMultiHeadAttention(MultiHeadAttention):
    """带掩码的多头注意力"""
    
    def forward(self, X, mask_type=None):
        """前向传播，支持掩码"""
        seq_len = X.shape[0]
        
        # 创建掩码
        if mask_type == 'causal':
            mask = create_causal_mask(seq_len)
        else:
            mask = None
        
        return super().forward(X, mask)


# 测试因果掩码
print("因果掩码示例 (seq_len=4):")
causal_mask = create_causal_mask(4)
print(causal_mask)
print("\n掩码说明:")
print("第1个位置只能看到自己")
print("第2个位置能看到前2个")
print("第3个位置能看到前3个")
print("第4个位置能看到全部")
```

---

### 8. Attention 的可视化与理解

#### 8.1 注意力权重可视化

```python
import matplotlib.pyplot as plt

def visualize_attention(weights, tokens, title="Attention Weights"):
    """可视化注意力权重"""
    plt.figure(figsize=(8, 6))
    plt.imshow(weights, cmap='Blues')
    
    plt.xticks(range(len(tokens)), tokens, fontsize=12)
    plt.yticks(range(len(tokens)), tokens, fontsize=12)
    
    plt.xlabel('Key', fontsize=14)
    plt.ylabel('Query', fontsize=14)
    plt.title(title, fontsize=16)
    
    # 显示数值
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            plt.text(j, i, f'{weights[i, j]:.2f}', 
                    ha='center', va='center', fontsize=10)
    
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('attention_weights.png', dpi=150)
    plt.show()


# 示例
np.random.seed(42)
tokens = ['我', '喜欢', '学习', 'AI']

# 模拟注意力权重
weights = np.array([
    [0.5, 0.2, 0.2, 0.1],
    [0.1, 0.6, 0.2, 0.1],
    [0.1, 0.2, 0.5, 0.2],
    [0.1, 0.1, 0.2, 0.6]
])

print("注意力权重矩阵:")
print(weights)
# visualize_attention(weights, tokens)
```

#### 8.2 理解注意力模式

```
常见注意力模式：

1. 对角线模式：
   - 每个位置主要关注自己
   - 适合保留局部信息

2. 均匀模式：
   - 所有权重相近
   - 类似于平均池化

3. 特定关注模式：
   - 某些位置关注特定其他位置
   - 例如：动词关注主语

4. 头的多样性：
   - 不同头学习不同模式
   - 有助于捕捉多种关系
```

---

## 💻 完整代码示例

### 示例：用 Attention 实现简单的序列分类

```python
"""
完整示例：使用 Self-Attention 进行文本分类
"""
import numpy as np

class AttentionClassifier:
    """基于 Attention 的文本分类器"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # 词嵌入
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        # Self-Attention
        self.W_q = np.random.randn(embedding_dim, hidden_size) * 0.1
        self.W_k = np.random.randn(embedding_dim, hidden_size) * 0.1
        self.W_v = np.random.randn(embedding_dim, hidden_size) * 0.1
        
        # 分类器
        self.W_c = np.random.randn(hidden_size, num_classes) * 0.1
        self.b_c = np.zeros(num_classes)
    
    def attention(self, X):
        """计算 self-attention"""
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        d_k = self.hidden_size
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        weights = self._softmax(scores)
        output = np.dot(weights, V)
        
        return output, weights
    
    def forward(self, input_ids):
        """前向传播"""
        # 嵌入
        X = self.embedding[input_ids]  # (seq_len, embedding_dim)
        
        # Self-Attention
        attn_output, weights = self.attention(X)  # (seq_len, hidden_size)
        
        # 池化（取平均）
        pooled = np.mean(attn_output, axis=0)  # (hidden_size,)
        
        # 分类
        logits = np.dot(pooled, self.W_c) + self.b_c
        probs = self._softmax(logits)
        
        return probs, weights
    
    def _softmax(self, x):
        if x.ndim == 1:
            e_x = np.exp(x - np.max(x))
            return e_x / np.sum(e_x)
        else:
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def train(self, data, labels, epochs=50, lr=0.01):
        """训练（简化版）"""
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            
            for input_ids, label in zip(data, labels):
                # 前向传播
                probs, _ = self.forward(input_ids)
                
                # 计算损失
                loss = -np.log(probs[label] + 1e-10)
                total_loss += loss
                
                # 预测
                if np.argmax(probs) == label:
                    correct += 1
                
                # 反向传播（简化：只更新分类器）
                grad = probs.copy()
                grad[label] -= 1
                
                # 获取池化输出
                X = self.embedding[input_ids]
                attn_output, _ = self.attention(X)
                pooled = np.mean(attn_output, axis=0)
                
                # 更新分类器
                self.W_c -= lr * np.outer(pooled, grad)
                self.b_c -= lr * grad
            
            if (epoch + 1) % 10 == 0:
                acc = correct / len(data)
                print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Acc: {acc:.4f}")


# 示例：情感分类
np.random.seed(42)

# 简单的词汇表
vocab = {"好": 0, "棒": 1, "喜欢": 2, "差": 3, "烂": 4, "讨厌": 5, "电影": 6, "很": 7}

# 训练数据
data = [
    [0, 7, 6],   # 好 很 电影 -> 正面
    [1, 6],      # 棒 电影 -> 正面
    [2, 6],      # 喜欢 电影 -> 正面
    [3, 7, 6],   # 差 很 电影 -> 负面
    [4, 6],      # 烂 电影 -> 负面
    [5, 6],      # 讨厌 电影 -> 负面
]
labels = [0, 0, 0, 1, 1, 1]  # 0=正面, 1=负面

# 创建并训练模型
classifier = AttentionClassifier(
    vocab_size=len(vocab),
    embedding_dim=16,
    hidden_size=16,
    num_classes=2
)

classifier.train(data, labels, epochs=100, lr=0.1)

# 测试
test_data = [
    [0, 6],   # 好 电影
    [4, 6],   # 烂 电影
]

print("\n测试结果:")
for ids in test_data:
    probs, weights = classifier.forward(ids)
    pred = "正面" if np.argmax(probs) == 0 else "负面"
    words = [list(vocab.keys())[i] for i in ids]
    print(f"输入: {words} -> 预测: {pred} (概率: {probs})")
```

---

## 🎯 实践练习

### 练习 1：实现 Cross-Attention

**任务**：实现编码器-解码器之间的 Cross-Attention。

```python
# 提示：Query 来自解码器，Key/Value 来自编码器
class CrossAttention:
    def __init__(self, decoder_dim, encoder_dim, hidden_size):
        # TODO: 实现
        pass
```

### 练习 2：实现 Local Attention

**任务**：实现只关注局部窗口的注意力，减少计算量。

### 练习 3：可视化 BERT 注意力

**任务**：加载预训练 BERT 模型，可视化不同层的注意力模式。

---

## 📝 本章小结

### 核心要点

1. **Attention 动机**：解决 Seq2Seq 的信息瓶颈问题
2. **Q/K/V 框架**：Query 查询 Key，获取相关 Value
3. **Self-Attention**：序列内部的注意力，每个位置与其他位置交互
4. **Multi-Head**：多个注意力头学习不同的关注模式
5. **Mask**：Padding Mask 和 Causal Mask

### 关键公式

```
Scaled Dot-Product Attention:
Attention(Q, K, V) = softmax(Q · K^T / √dₖ) · V

Multi-Head Attention:
MultiHead(Q, K, V) = Concat(head₁, ..., head_h) · W_O
where head_i = Attention(Q · W_i^Q, K · W_i^K, V · W_i^V)

Bahdanau Attention:
score(s_t, h_i) = v^T · tanh(Ws · s_t + Wh · h_i)
```

### Attention 类型对比

| 类型 | Query 来源 | Key/Value 来源 | 应用场景 |
|------|-----------|---------------|----------|
| Self-Attention | 输入序列 | 输入序列 | Transformer 编码器 |
| Cross-Attention | 解码器 | 编码器 | Seq2Seq 解码器 |
| Masked Self-Attention | 输入序列 | 输入序列（掩码） | Transformer 解码器 |

---

<div align="center">

[⬅️ 上一章](../chapter06-rnn/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter08-transformer/README.md)

</div>
