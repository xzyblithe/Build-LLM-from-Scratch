# 第8章：Transformer 详解

<div align="center">

[⬅️ 上一章](../chapter07-attention/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter09-pretraining/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 深入理解 Transformer 的整体架构
- ✅ 掌握编码器和解码器的结构
- ✅ 理解位置编码的原理与实现
- ✅ 掌握 Layer Normalization 和残差连接
- ✅ 理解前馈网络（FFN）的作用
- ✅ 从零实现完整的 Transformer 模型
- ✅ 理解 Transformer 的训练技巧

---

## 🎯 本章内容

### 1. Transformer 概述

#### 1.1 历史背景

Transformer 由 Google 在 2017 年的论文《Attention Is All You Need》中提出，彻底改变了 NLP 领域。

```
发展历程：

RNN/LSTM (2015)
    ↓ 引入 Attention
RNN + Attention (2015-2016)
    ↓ 纯 Attention
Transformer (2017) ← 里程碑
    ↓ 预训练
BERT, GPT (2018-)
```

#### 1.2 Transformer 的优势

```
相比 RNN：

1. 并行计算
   - RNN 必须顺序处理，无法并行
   - Transformer 所有位置同时计算，训练速度快

2. 长距离依赖
   - RNN 需要逐步传递信息，距离越远越难学习
   - Transformer 任意两个位置直接相连

3. 梯度流动
   - RNN 容易梯度消失
   - Transformer 残差连接使梯度流动更顺畅
```

#### 1.3 整体架构

```
Transformer 架构：

输入 → [Embedding + Position Encoding]
         ↓
┌─────────────────────────────────────┐
│         Encoder (× N 层)             │
│  ┌─────────────────────────────┐    │
│  │ Multi-Head Self-Attention   │    │
│  │ Add & Norm                  │    │
│  ├─────────────────────────────┤    │
│  │ Feed Forward Network        │    │
│  │ Add & Norm                  │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
         ↓ 编码器输出
┌─────────────────────────────────────┐
│         Decoder (× N 层)             │
│  ┌─────────────────────────────┐    │
│  │ Masked Self-Attention       │    │
│  │ Add & Norm                  │    │
│  ├─────────────────────────────┤    │
│  │ Cross-Attention             │    │
│  │ Add & Norm                  │    │
│  ├─────────────────────────────┤    │
│  │ Feed Forward Network        │    │
│  │ Add & Norm                  │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
         ↓
    [Linear + Softmax]
         ↓
       输出概率
```

---

### 2. 输入嵌入与位置编码

#### 2.1 词嵌入（Token Embedding）

```
将词转换为向量：

输入: 词索引序列 [w₁, w₂, ..., wₙ]
嵌入矩阵: E ∈ R^(vocab_size × d_model)
输出: X = [E[w₁], E[w₂], ..., E[wₙ]]

其中 d_model 是模型维度（论文中为 512）
```

#### 2.2 为什么需要位置编码？

```
问题：
Self-Attention 对位置不敏感

Attention(Q, K, V) 对序列打乱后结果相同（除了位置交换）
模型不知道词的顺序！

例子：
"我喜欢你" 和 "你喜欢我" 的 Self-Attention 输出结构相同
但意思完全不同！
```

#### 2.3 正弦位置编码

Transformer 使用正弦和余弦函数生成位置编码：

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中：
- pos: 位置索引
- i: 维度索引
- d_model: 模型维度
```

**设计优点**：
1. 每个位置有唯一的编码
2. 不同位置之间的编码可以通过线性变换得到
3. 可以泛化到任意长度

```python
import numpy as np

def positional_encoding(max_len, d_model):
    """
    生成位置编码
    
    参数:
        max_len: 最大序列长度
        d_model: 模型维度
    
    返回:
        PE: shape (max_len, d_model)
    """
    PE = np.zeros((max_len, d_model))
    
    for pos in range(max_len):
        for i in range(d_model):
            if i % 2 == 0:
                PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            else:
                PE[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
    
    return PE

# 向量化实现（更高效）
def positional_encoding_vectorized(max_len, d_model):
    """向量化位置编码实现"""
    position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    PE = np.zeros((max_len, d_model))
    PE[:, 0::2] = np.sin(position * div_term)  # 偶数维度
    PE[:, 1::2] = np.cos(position * div_term)  # 奇数维度
    
    return PE


# 测试
PE = positional_encoding_vectorized(100, 512)
print("位置编码形状:", PE.shape)
print("\n位置 0 的前 10 维:")
print(PE[0, :10])
print("\n位置 1 的前 10 维:")
print(PE[1, :10])
```

#### 2.4 可视化位置编码

```python
import matplotlib.pyplot as plt

def visualize_positional_encoding(PE):
    """可视化位置编码"""
    plt.figure(figsize=(12, 6))
    plt.imshow(PE.T, aspect='auto', cmap='RdBu')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('positional_encoding.png', dpi=150)
    plt.show()

# PE = positional_encoding_vectorized(100, 128)
# visualize_positional_encoding(PE)
```

---

### 3. Transformer 编码器

#### 3.1 多头自注意力层

```python
class MultiHeadSelfAttention:
    """多头自注意力"""
    
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Q, K, V 投影
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        
        # 输出投影
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, X, mask=None):
        """
        前向传播
        
        参数:
            X: shape (seq_len, d_model)
            mask: 可选掩码
        
        返回:
            output: shape (seq_len, d_model)
        """
        seq_len = X.shape[0]
        
        # 线性投影
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # 重塑为多头
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        
        # 计算注意力
        outputs = []
        for h in range(self.num_heads):
            # 缩放点积注意力
            scores = np.dot(Q[h], K[h].T) / np.sqrt(self.head_dim)
            
            if mask is not None:
                scores += mask * -1e9
            
            weights = self._softmax(scores)
            output = np.dot(weights, V[h])
            outputs.append(output)
        
        # 拼接并投影
        concat = np.concatenate(outputs, axis=1)
        output = np.dot(concat, self.W_o)
        
        return output
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
```

#### 3.2 前馈网络（FFN）

```
FFN 结构：

FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

即：两个线性变换 + ReLU 激活

维度变化：
d_model → d_ff → d_model

论文中 d_ff = 2048, d_model = 512
```

```python
class FeedForward:
    """前馈网络"""
    
    def __init__(self, d_model, d_ff):
        """
        参数:
            d_model: 模型维度
            d_ff: 隐藏层维度
        """
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: shape (seq_len, d_model)
        
        返回:
            output: shape (seq_len, d_model)
        """
        # 第一层 + ReLU
        hidden = np.maximum(0, np.dot(X, self.W1) + self.b1)
        
        # 第二层
        output = np.dot(hidden, self.W2) + self.b2
        
        return output
```

#### 3.3 层归一化（Layer Normalization）

```
Batch Normalization vs Layer Normalization：

Batch Norm: 对每个特征在 batch 维度上归一化
Layer Norm: 对每个样本在特征维度上归一化

Transformer 使用 Layer Norm，因为：
1. 不依赖 batch size
2. 序列长度可变时更稳定
```

```python
class LayerNorm:
    """层归一化"""
    
    def __init__(self, d_model, eps=1e-6):
        """
        参数:
            d_model: 特征维度
            eps: 防止除零的小常数
        """
        self.eps = eps
        self.gamma = np.ones(d_model)  # 可学习参数
        self.beta = np.zeros(d_model)  # 可学习参数
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: shape (seq_len, d_model) 或 (batch, seq_len, d_model)
        
        返回:
            归一化后的输出
        """
        # 计算均值和方差
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        
        # 归一化
        X_norm = (X - mean) / np.sqrt(var + self.eps)
        
        # 缩放和平移
        return self.gamma * X_norm + self.beta
```

#### 3.4 残差连接（Residual Connection）

```
残差连接：

output = LayerNorm(x + Sublayer(x))

作用：
1. 缓解梯度消失
2. 允许模型学习恒等映射
3. 加速训练收敛
```

```python
class SublayerConnection:
    """残差连接 + 层归一化"""
    
    def __init__(self, d_model):
        self.norm = LayerNorm(d_model)
    
    def forward(self, X, sublayer_output):
        """
        前向传播
        
        参数:
            X: 原始输入
            sublayer_output: 子层输出
        
        返回:
            残差连接后的输出
        """
        return self.norm(X + sublayer_output)
```

#### 3.5 完整的编码器层

```python
class EncoderLayer:
    """Transformer 编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏维度
        """
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, X, mask=None):
        """
        前向传播
        
        参数:
            X: shape (seq_len, d_model)
            mask: 可选掩码
        
        返回:
            output: shape (seq_len, d_model)
        """
        # 自注意力 + 残差 + 归一化
        attn_out = self.self_attn.forward(X, mask)
        X = self.norm1(X + attn_out)
        
        # FFN + 残差 + 归一化
        ffn_out = self.ffn.forward(X)
        X = self.norm2(X + ffn_out)
        
        return X


class TransformerEncoder:
    """Transformer 编码器"""
    
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        """
        参数:
            num_layers: 层数
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏维度
        """
        self.layers = [EncoderLayer(d_model, num_heads, d_ff) 
                       for _ in range(num_layers)]
        self.norm = LayerNorm(d_model)
    
    def forward(self, X, mask=None):
        """前向传播"""
        for layer in self.layers:
            X = layer.forward(X, mask)
        
        return self.norm(X)


# 测试编码器
np.random.seed(42)

d_model = 64
num_heads = 8
d_ff = 256
num_layers = 2
seq_len = 10

# 创建编码器
encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)

# 模拟输入
X = np.random.randn(seq_len, d_model)

# 前向传播
output = encoder.forward(X)

print("输入形状:", X.shape)
print("输出形状:", output.shape)
```

---

### 4. Transformer 解码器

#### 4.1 解码器结构

解码器比编码器多了一个 Cross-Attention 层：

```
解码器层结构：

1. Masked Self-Attention
   - 使用因果掩码，防止看到未来信息
   
2. Cross-Attention (Encoder-Decoder Attention)
   - Query 来自解码器
   - Key/Value 来自编码器输出
   
3. Feed Forward Network
```

#### 4.2 实现

```python
class DecoderLayer:
    """Transformer 解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model
        
        # Masked Self-Attention
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        
        # Cross-Attention
        self.cross_attn = MultiHeadSelfAttention(d_model, num_heads)
        
        # FFN
        self.ffn = FeedForward(d_model, d_ff)
        
        # Layer Norms
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, X, encoder_output, self_mask=None, cross_mask=None):
        """
        前向传播
        
        参数:
            X: 解码器输入, shape (tgt_len, d_model)
            encoder_output: 编码器输出, shape (src_len, d_model)
            self_mask: 自注意力掩码（因果掩码）
            cross_mask: 交叉注意力掩码
        """
        # Masked Self-Attention
        self_attn_out = self.self_attn.forward(X, self_mask)
        X = self.norm1(X + self_attn_out)
        
        # Cross-Attention
        # Query 来自解码器，Key/Value 来自编码器
        cross_attn_out = self._cross_attention(X, encoder_output, cross_mask)
        X = self.norm2(X + cross_attn_out)
        
        # FFN
        ffn_out = self.ffn.forward(X)
        X = self.norm3(X + ffn_out)
        
        return X
    
    def _cross_attention(self, X, encoder_output, mask=None):
        """交叉注意力"""
        # 简化实现：使用 encoder_output 作为 K 和 V
        seq_len = X.shape[0]
        src_len = encoder_output.shape[0]
        
        # Q 来自解码器，K/V 来自编码器
        Q = np.dot(X, self.cross_attn.W_q)
        K = np.dot(encoder_output, self.cross_attn.W_k)
        V = np.dot(encoder_output, self.cross_attn.W_v)
        
        # 多头处理
        outputs = []
        for h in range(self.cross_attn.num_heads):
            Q_h = Q[:, h*self.cross_attn.head_dim:(h+1)*self.cross_attn.head_dim]
            K_h = K[:, h*self.cross_attn.head_dim:(h+1)*self.cross_attn.head_dim]
            V_h = V[:, h*self.cross_attn.head_dim:(h+1)*self.cross_attn.head_dim]
            
            scores = np.dot(Q_h, K_h.T) / np.sqrt(self.cross_attn.head_dim)
            
            if mask is not None:
                scores += mask * -1e9
            
            weights = self._softmax(scores)
            output = np.dot(weights, V_h)
            outputs.append(output)
        
        concat = np.concatenate(outputs, axis=1)
        return np.dot(concat, self.cross_attn.W_o)
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


class TransformerDecoder:
    """Transformer 解码器"""
    
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        self.layers = [DecoderLayer(d_model, num_heads, d_ff) 
                       for _ in range(num_layers)]
        self.norm = LayerNorm(d_model)
    
    def forward(self, X, encoder_output, self_mask=None, cross_mask=None):
        """前向传播"""
        for layer in self.layers:
            X = layer.forward(X, encoder_output, self_mask, cross_mask)
        
        return self.norm(X)
```

---

### 5. 完整的 Transformer

#### 5.1 组合所有组件

```python
class Transformer:
    """完整的 Transformer 模型"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, 
                 max_len=512):
        """
        参数:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 编码器/解码器层数
            d_ff: 前馈网络隐藏维度
            max_len: 最大序列长度
        """
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # 位置编码
        self.PE = positional_encoding_vectorized(max_len, d_model)
        
        # 编码器和解码器
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff)
        
        # 输出层
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.02
    
    def encode(self, src_ids, src_mask=None):
        """编码"""
        seq_len = len(src_ids)
        
        # 嵌入 + 位置编码
        X = self.embedding[src_ids] * np.sqrt(self.d_model)
        X = X + self.PE[:seq_len]
        
        # 编码器
        return self.encoder.forward(X, src_mask)
    
    def decode(self, tgt_ids, encoder_output, tgt_mask=None, src_mask=None):
        """解码"""
        seq_len = len(tgt_ids)
        
        # 嵌入 + 位置编码
        X = self.embedding[tgt_ids] * np.sqrt(self.d_model)
        X = X + self.PE[:seq_len]
        
        # 解码器
        return self.decoder.forward(X, encoder_output, tgt_mask, src_mask)
    
    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        """完整前向传播"""
        # 编码
        encoder_output = self.encode(src_ids, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt_ids, encoder_output, tgt_mask, src_mask)
        
        # 输出投影
        logits = np.dot(decoder_output, self.output_proj)
        
        return logits
    
    def generate(self, src_ids, max_len=50, bos_id=1, eos_id=2):
        """生成序列（贪婪解码）"""
        # 编码
        encoder_output = self.encode(src_ids)
        
        # 初始化目标序列
        tgt_ids = [bos_id]
        
        for _ in range(max_len):
            # 创建因果掩码
            tgt_mask = self._create_causal_mask(len(tgt_ids))
            
            # 解码
            logits = self.decode(np.array(tgt_ids), encoder_output, tgt_mask)
            
            # 取最后一个位置的预测
            next_token = np.argmax(logits[-1])
            
            if next_token == eos_id:
                break
            
            tgt_ids.append(next_token)
        
        return tgt_ids
    
    def _create_causal_mask(self, seq_len):
        """创建因果掩码"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask


# 测试
np.random.seed(42)

vocab_size = 1000
d_model = 64
num_heads = 4
num_layers = 2
d_ff = 256

model = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff)

# 模拟输入
src_ids = np.random.randint(0, vocab_size, size=10)
tgt_ids = np.random.randint(0, vocab_size, size=8)

# 前向传播
logits = model.forward(src_ids, tgt_ids)

print("源序列长度:", len(src_ids))
print("目标序列长度:", len(tgt_ids))
print("输出 logits 形状:", logits.shape)
```

---

### 6. Transformer 的关键细节

#### 6.1 参数初始化

```python
def init_transformer_weights(d_model):
    """Xavier 初始化"""
    # 权重使用 Xavier/Glorot 初始化
    scale = np.sqrt(2.0 / d_model)
    
    # 实际应用中使用
    # W = np.random.randn(d_model, d_model) * scale
    
    # 或者使用更小的初始化（论文做法）
    scale = 0.02
    # W = np.random.randn(d_model, d_model) * scale
    
    return scale
```

#### 6.2 学习率调度

Transformer 使用特殊的学习率调度：

```
lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

先线性增加，后按步数的平方根衰减
```

```python
def get_learning_rate(step, d_model, warmup_steps=4000):
    """计算学习率"""
    arg1 = step ** (-0.5)
    arg2 = step * (warmup_steps ** (-1.5))
    
    return (d_model ** (-0.5)) * min(arg1, arg2)


# 可视化学习率变化
import matplotlib.pyplot as plt

d_model = 512
steps = range(1, 20000)
lrs = [get_learning_rate(s, d_model) for s in steps]

plt.figure(figsize=(10, 5))
plt.plot(steps, lrs)
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.title('Transformer Learning Rate Schedule')
plt.grid(True, alpha=0.3)
plt.savefig('learning_rate_schedule.png', dpi=150)
# plt.show()
```

#### 6.3 标签平滑

```python
def label_smoothing(targets, num_classes, smoothing=0.1):
    """
    标签平滑
    
    将 one-hot 标签变为平滑分布：
    y_smooth = (1-ε) * y_onehot + ε / num_classes
    """
    smooth_targets = np.zeros((len(targets), num_classes))
    
    for i, target in enumerate(targets):
        smooth_targets[i] = smoothing / (num_classes - 1)
        smooth_targets[i, target] = 1.0 - smoothing
    
    return smooth_targets


# 示例
targets = [1, 2, 0]
smooth = label_smoothing(targets, num_classes=5, smoothing=0.1)
print("原始标签:", targets)
print("\n平滑后的分布:")
print(smooth)
```

---

## 💻 完整代码示例

### 示例：简易机器翻译模型

```python
"""
完整示例：使用 Transformer 进行简单的序列到序列任务
"""
import numpy as np

class SimpleTransformer:
    """简化的 Transformer，用于演示"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, 
                 num_heads=4, num_layers=2, d_ff=128):
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # 嵌入层
        self.src_embedding = np.random.randn(src_vocab_size, d_model) * 0.02
        self.tgt_embedding = np.random.randn(tgt_vocab_size, d_model) * 0.02
        
        # 位置编码
        self.PE = positional_encoding_vectorized(100, d_model)
        
        # 编码器层
        self.encoder_layers = [
            {'W_q': np.random.randn(d_model, d_model) * 0.02,
             'W_k': np.random.randn(d_model, d_model) * 0.02,
             'W_v': np.random.randn(d_model, d_model) * 0.02,
             'W_o': np.random.randn(d_model, d_model) * 0.02,
             'W1': np.random.randn(d_model, d_ff) * 0.02,
             'b1': np.zeros(d_ff),
             'W2': np.random.randn(d_ff, d_model) * 0.02,
             'b2': np.zeros(d_model)}
            for _ in range(num_layers)
        ]
        
        # 输出层
        self.output_proj = np.random.randn(d_model, tgt_vocab_size) * 0.02
    
    def attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        d_k = Q.shape[-1]
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        
        if mask is not None:
            scores += mask * -1e9
        
        weights = self._softmax(scores)
        return np.dot(weights, V), weights
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def layer_norm(self, X, eps=1e-6):
        """层归一化"""
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        return (X - mean) / np.sqrt(var + eps)
    
    def encode(self, src_ids):
        """编码"""
        X = self.src_embedding[src_ids] * np.sqrt(self.d_model)
        X = X + self.PE[:len(src_ids)]
        
        for layer in self.encoder_layers:
            # Self-Attention
            Q = np.dot(X, layer['W_q'])
            K = np.dot(X, layer['W_k'])
            V = np.dot(X, layer['W_v'])
            attn_out, _ = self.attention(Q, K, V)
            attn_out = np.dot(attn_out, layer['W_o'])
            X = self.layer_norm(X + attn_out)
            
            # FFN
            ffn_out = np.maximum(0, np.dot(X, layer['W1']) + layer['b1'])
            ffn_out = np.dot(ffn_out, layer['W2']) + layer['b2']
            X = self.layer_norm(X + ffn_out)
        
        return X
    
    def forward(self, src_ids, tgt_ids):
        """前向传播"""
        encoder_output = self.encode(src_ids)
        
        # 解码器（简化）
        X = self.tgt_embedding[tgt_ids] * np.sqrt(self.d_model)
        X = X + self.PE[:len(tgt_ids)]
        
        # 简化的解码器（单层）
        for layer in self.encoder_layers:
            Q = np.dot(X, layer['W_q'])
            K = np.dot(encoder_output, layer['W_k'])
            V = np.dot(encoder_output, layer['W_v'])
            attn_out, _ = self.attention(Q, K, V)
            attn_out = np.dot(attn_out, layer['W_o'])
            X = self.layer_norm(X + attn_out)
        
        logits = np.dot(X, self.output_proj)
        return logits
    
    def translate(self, src_ids, max_len=20, bos_id=1, eos_id=2):
        """翻译"""
        encoder_output = self.encode(src_ids)
        tgt_ids = [bos_id]
        
        for _ in range(max_len):
            logits = self.forward(src_ids, np.array(tgt_ids))
            next_token = np.argmax(logits[-1])
            
            if next_token == eos_id:
                break
            
            tgt_ids.append(next_token)
        
        return tgt_ids[1:]  # 去掉 BOS


# 演示
np.random.seed(42)

model = SimpleTransformer(
    src_vocab_size=100,
    tgt_vocab_size=100,
    d_model=32,
    num_heads=2,
    num_layers=1
)

src = np.array([10, 20, 30, 40, 0, 0])  # 源序列 + padding
tgt = np.array([1, 50, 60, 70, 80])     # 目标序列

logits = model.forward(src, tgt)
print("输入形状:", src.shape)
print("输出形状:", logits.shape)
print("\n每个位置预测的词:")
print(np.argmax(logits, axis=-1))
```

---

## 🎯 实践练习

### 练习 1：实现 Beam Search

**任务**：实现束搜索解码，代替贪婪解码。

```python
def beam_search(model, src_ids, beam_size=4, max_len=50):
    """
    束搜索解码
    
    提示：
    1. 维护 beam_size 个候选序列
    2. 每步扩展所有候选，保留 top-k
    3. 返回概率最高的序列
    """
    # TODO: 实现
    pass
```

### 练习 2：添加 Dropout

**任务**：在 Transformer 中添加 Dropout 正则化。

### 练习 3：实现 BERT 风格的 Encoder-Only

**任务**：只使用编码器部分，实现文本分类。

---

## 📝 本章小结

### 核心要点

1. **整体架构**：编码器-解码器结构，N 层堆叠
2. **位置编码**：正弦函数，让模型感知位置
3. **多头注意力**：多角度关注序列
4. **前馈网络**：两层线性变换 + ReLU
5. **残差连接 + 层归一化**：稳定训练

### 关键公式

```
位置编码:
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

多头注意力:
MultiHead(Q,K,V) = Concat(head₁,...,head_h)W_O

前馈网络:
FFN(x) = max(0, xW₁+b₁)W₂+b₂

残差连接:
output = LayerNorm(x + Sublayer(x))
```

### Transformer 参数规模

```
Base 模型（论文）：
- d_model = 512
- num_heads = 8
- num_layers = 6
- d_ff = 2048
- 参数量：约 65M

Large 模型：
- d_model = 1024
- num_heads = 16
- num_layers = 12
- d_ff = 4096
- 参数量：约 213M
```

---

<div align="center">

[⬅️ 上一章](../chapter07-attention/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter09-pretraining/README.md)

</div>
