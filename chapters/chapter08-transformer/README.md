# 第8章：Transformer 详解 ⭐

<div align="center">

[⬅️ 上一章](../chapter07-attention/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter09-pretraining/README.md)

**🎯 本章是理解大语言模型的核心基础**

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 深入理解 Transformer 的整体架构
- ✅ 掌握 Self-Attention 和 Multi-Head Attention
- ✅ 理解位置编码（Positional Encoding）
- ✅ 掌握 Encoder 和 Decoder 的结构
- ✅ 从零实现一个完整的 Transformer

---

## 🎯 本章内容

### 1. Transformer 概述

#### 1.1 Transformer 的诞生

Transformer 由 Google 在 2017 年论文《Attention Is All You Need》中提出，彻底改变了 NLP 领域。

**核心创新**：
- 完全基于注意力机制，摒弃了 RNN 和 CNN
- 并行计算能力强
- 能够捕捉长距离依赖

**影响**：
- BERT、GPT、LLaMA 等大模型都基于 Transformer
- 成为 NLP 领域的标准架构

---

#### 1.2 整体架构

```
Transformer 架构
┌─────────────────────────────────┐
│         输出层                   │
├─────────────────────────────────┤
│    Decoder (N 层)               │
│  ┌─────────────────────────┐   │
│  │ Masked Self-Attention   │   │
│  │ Cross-Attention         │   │
│  │ Feed Forward            │   │
│  └─────────────────────────┘   │
├─────────────────────────────────┤
│    Encoder (N 层)               │
│  ┌─────────────────────────┐   │
│  │ Multi-Head Attention    │   │
│  │ Feed Forward            │   │
│  └─────────────────────────┘   │
├─────────────────────────────────┤
│    输入嵌入 + 位置编码         │
└─────────────────────────────────┘
```

---

### 2. 核心组件详解

#### 2.1 Self-Attention 机制

Self-Attention 是 Transformer 的核心，让每个词都能关注到序列中的所有词。

**计算流程**：

```
输入: X ∈ R^(n×d)
Q = X·W_Q  (Query)
K = X·W_K  (Key)
V = X·W_V  (Value)

Attention(Q, K, V) = softmax(Q·K^T / √d_k)·V
```

```python
import numpy as np

def self_attention(X, W_Q, W_K, W_V):
    """
    Self-Attention 实现
    
    参数:
        X: 输入序列 (seq_len, d_model)
        W_Q, W_K, W_V: 权重矩阵
    
    返回:
        注意力输出
    """
    d_k = W_Q.shape[1]
    
    # 计算 Q, K, V
    Q = np.dot(X, W_Q)  # (seq_len, d_k)
    K = np.dot(X, W_K)  # (seq_len, d_k)
    V = np.dot(X, W_V)  # (seq_len, d_v)
    
    # 计算注意力分数
    scores = np.dot(Q, K.T) / np.sqrt(d_k)  # (seq_len, seq_len)
    
    # Softmax
    attention_weights = softmax(scores)  # (seq_len, seq_len)
    
    # 加权求和
    output = np.dot(attention_weights, V)  # (seq_len, d_v)
    
    return output, attention_weights

def softmax(x):
    """Softmax 函数"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# 示例
np.random.seed(42)
X = np.random.randn(3, 4)  # 序列长度3，维度4
W_Q = np.random.randn(4, 3)
W_K = np.random.randn(4, 3)
W_V = np.random.randn(4, 3)

output, weights = self_attention(X, W_Q, W_K, W_V)

print("注意力权重矩阵:")
print(weights.round(4))
print("\n输出:")
print(output.round(4))
```

**直观理解**：

```
句子: "我 爱 北京 天安门"

Q: 每个词在"寻找"什么
K: 每个词"提供"的信息
V: 每个词的实际内容

注意力权重: "爱" 关注 "我" 和 "北京"
            "天安门" 关注 "北京"
```

---

#### 2.2 Multi-Head Attention

多头注意力让模型可以同时关注不同的表示子空间。

```python
class MultiHeadAttention:
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads):
        """
        参数:
            d_model: 模型维度
            num_heads: 头的数量
        """
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 初始化权重
        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
        self.W_O = np.random.randn(d_model, d_model) * 0.01
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        d_k = Q.shape[-1]
        
        # 计算注意力分数
        scores = np.dot(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax
        attention_weights = softmax(scores)
        
        # 加权求和
        output = np.dot(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, X, mask=None):
        """
        前向传播
        
        参数:
            X: 输入 (batch_size, seq_len, d_model)
            mask: 掩码
        
        返回:
            输出, 注意力权重
        """
        batch_size = X.shape[0]
        
        # 线性变换
        Q = np.dot(X, self.W_Q)  # (batch_size, seq_len, d_model)
        K = np.dot(X, self.W_K)
        V = np.dot(X, self.W_V)
        
        # 分割成多头
        Q = Q.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 注意力计算
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )
        
        # 拼接多头
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(
            batch_size, -1, self.d_model
        )
        
        # 输出投影
        output = np.dot(attention_output, self.W_O)
        
        return output, attention_weights

# 示例
d_model = 512
num_heads = 8
seq_len = 10
batch_size = 2

X = np.random.randn(batch_size, seq_len, d_model)
mha = MultiHeadAttention(d_model, num_heads)
output, weights = mha.forward(X)

print(f"输入形状: {X.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")
```

**多头的作用**：

```
头1: 关注语法结构
头2: 关注语义相似性
头3: 关注指代关系
...

每个头学习不同的表示子空间
```

---

#### 2.3 Positional Encoding

由于 Transformer 没有循环结构，需要位置编码来注入位置信息。

**正弦余弦编码**：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

```python
def positional_encoding(seq_len, d_model):
    """
    位置编码
    
    参数:
        seq_len: 序列长度
        d_model: 模型维度
    
    返回:
        位置编码矩阵 (seq_len, d_model)
    """
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # 偶数维度
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数维度
    
    return pe

# 示例
seq_len = 50
d_model = 512

pe = positional_encoding(seq_len, d_model)

print(f"位置编码形状: {pe.shape}")
print(f"\n第1个位置的编码（前10维）:")
print(pe[0, :10].round(4))

# 可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.imshow(pe.T, aspect='auto', cmap='viridis')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Positional Encoding')
plt.colorbar()
plt.tight_layout()
plt.savefig('positional_encoding.png', dpi=300, bbox_inches='tight')
print("\n位置编码可视化已保存为 positional_encoding.png")
plt.show()
```

**位置编码的特点**：
- 每个位置有唯一的编码
- 不同位置之间的距离可以通过编码间的距离体现
- 可以推广到任意长度的序列

---

#### 2.4 Feed-Forward Network

每个 Encoder/Decoder 层都包含一个前馈网络。

```python
class FeedForward:
    """前馈网络"""
    
    def __init__(self, d_model, d_ff):
        """
        参数:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度
        """
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入 (batch_size, seq_len, d_model)
        
        返回:
            输出 (batch_size, seq_len, d_model)
        """
        # 第一层 + ReLU
        hidden = np.maximum(0, np.dot(X, self.W1) + self.b1)
        
        # 第二层
        output = np.dot(hidden, self.W2) + self.b2
        
        return output

# 示例
d_model = 512
d_ff = 2048

ff = FeedForward(d_model, d_ff)
X = np.random.randn(2, 10, d_model)
output = ff.forward(X)

print(f"前馈网络输出形状: {output.shape}")
```

---

#### 2.5 Layer Normalization

Layer Normalization 用于稳定训练。

```python
class LayerNorm:
    """层归一化"""
    
    def __init__(self, d_model, epsilon=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.epsilon = epsilon
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入 (batch_size, seq_len, d_model)
        
        返回:
            归一化后的输出
        """
        # 计算均值和方差
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        
        # 归一化
        X_norm = (X - mean) / np.sqrt(var + self.epsilon)
        
        # 缩放和平移
        output = self.gamma * X_norm + self.beta
        
        return output
```

---

### 3. Encoder 层

```python
class EncoderLayer:
    """Encoder 单层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout
    
    def dropout_layer(self, X):
        """Dropout"""
        mask = (np.random.rand(*X.shape) > self.dropout) / (1 - self.dropout)
        return X * mask
    
    def forward(self, X, mask=None):
        """
        前向传播
        
        参数:
            X: 输入 (batch_size, seq_len, d_model)
            mask: 掩码
        
        返回:
            输出
        """
        # Multi-Head Attention
        mha_output, _ = self.mha.forward(X, mask)
        mha_output = self.dropout_layer(mha_output)
        
        # Add & Norm
        X = self.norm1.forward(X + mha_output)
        
        # Feed Forward
        ff_output = self.ff.forward(X)
        ff_output = self.dropout_layer(ff_output)
        
        # Add & Norm
        X = self.norm2.forward(X + ff_output)
        
        return X
```

---

### 4. Decoder 层

```python
class DecoderLayer:
    """Decoder 单层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.mha1 = MultiHeadAttention(d_model, num_heads)  # Self-Attention
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # Cross-Attention
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = dropout
    
    def forward(self, X, encoder_output, look_ahead_mask=None, padding_mask=None):
        """
        前向传播
        
        参数:
            X: Decoder 输入
            encoder_output: Encoder 输出
            look_ahead_mask: 前瞻掩码
            padding_mask: 填充掩码
        
        返回:
            输出
        """
        # Masked Self-Attention
        mha1_output, _ = self.mha1.forward(X, look_ahead_mask)
        X = self.norm1.forward(X + mha1_output)
        
        # Cross-Attention
        mha2_output, _ = self.mha2.forward(
            X, encoder_output, encoder_output, padding_mask
        )
        X = self.norm2.forward(X + mha2_output)
        
        # Feed Forward
        ff_output = self.ff.forward(X)
        X = self.norm3.forward(X + ff_output)
        
        return X
```

---

## 💻 完整 Transformer 实现

```python
"""
完整 Transformer 模型实现
"""
import numpy as np

class Transformer:
    """完整 Transformer 模型"""
    
    def __init__(self, 
                 num_layers=6,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048,
                 input_vocab_size=10000,
                 target_vocab_size=10000,
                 max_seq_len=512,
                 dropout=0.1):
        """
        初始化 Transformer
        
        参数:
            num_layers: Encoder/Decoder 层数
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络维度
            input_vocab_size: 输入词汇表大小
            target_vocab_size: 目标词汇表大小
            max_seq_len: 最大序列长度
            dropout: Dropout 率
        """
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embedding 层
        self.encoder_embedding = np.random.randn(input_vocab_size, d_model) * 0.01
        self.decoder_embedding = np.random.randn(target_vocab_size, d_model) * 0.01
        
        # 位置编码
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        
        # Encoder 层
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        
        # Decoder 层
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        
        # 输出层
        self.output_projection = np.random.randn(d_model, target_vocab_size) * 0.01
    
    def encode(self, X, mask=None):
        """Encoder 前向传播"""
        seq_len = X.shape[1]
        
        # Embedding + Positional Encoding
        X = self.encoder_embedding[X] + self.pos_encoding[:seq_len]
        
        # Encoder 层
        for layer in self.encoder_layers:
            X = layer.forward(X, mask)
        
        return X
    
    def decode(self, X, encoder_output, look_ahead_mask=None, padding_mask=None):
        """Decoder 前向传播"""
        seq_len = X.shape[1]
        
        # Embedding + Positional Encoding
        X = self.decoder_embedding[X] + self.pos_encoding[:seq_len]
        
        # Decoder 层
        for layer in self.decoder_layers:
            X = layer.forward(X, encoder_output, look_ahead_mask, padding_mask)
        
        return X
    
    def forward(self, encoder_input, decoder_input, 
                encoder_mask=None, decoder_mask=None):
        """
        完整前向传播
        
        参数:
            encoder_input: Encoder 输入 (batch_size, enc_seq_len)
            decoder_input: Decoder 输入 (batch_size, dec_seq_len)
            encoder_mask: Encoder 掩码
            decoder_mask: Decoder 掩码
        
        返回:
            输出概率分布
        """
        # Encoder
        encoder_output = self.encode(encoder_input, encoder_mask)
        
        # Decoder
        decoder_output = self.decode(
            decoder_input, encoder_output, decoder_mask, encoder_mask
        )
        
        # 输出投影
        logits = np.dot(decoder_output, self.output_projection)
        
        # Softmax
        output = softmax(logits)
        
        return output
    
    def generate(self, encoder_input, start_token, max_length=50):
        """
        生成序列（推理）
        
        参数:
            encoder_input: Encoder 输入
            start_token: 起始 token
            max_length: 最大生成长度
        
        返回:
            生成的序列
        """
        # Encode
        encoder_output = self.encode(encoder_input)
        
        # 初始化 decoder 输入
        decoder_input = np.array([[start_token]])
        
        generated = [start_token]
        
        for _ in range(max_length):
            # Decode
            decoder_output = self.decode(decoder_input, encoder_output)
            
            # 获取最后一个位置的输出
            logits = np.dot(decoder_output[:, -1, :], self.output_projection)
            
            # Greedy decoding
            next_token = np.argmax(logits, axis=-1)[0]
            
            generated.append(next_token)
            
            # 更新 decoder 输入
            decoder_input = np.concatenate([
                decoder_input, 
                np.array([[next_token]])
            ], axis=1)
            
            # 如果生成了结束符，停止
            if next_token == 2:  # 假设 2 是结束符
                break
        
        return generated

# 示例使用
print("初始化 Transformer 模型...")
transformer = Transformer(
    num_layers=6,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    input_vocab_size=10000,
    target_vocab_size=10000
)

print(f"模型参数:")
print(f"  层数: {transformer.num_layers}")
print(f"  模型维度: {transformer.d_model}")
print(f"  注意力头数: 8")

# 前向传播测试
batch_size = 2
enc_seq_len = 10
dec_seq_len = 8

encoder_input = np.random.randint(0, 10000, (batch_size, enc_seq_len))
decoder_input = np.random.randint(0, 10000, (batch_size, dec_seq_len))

print(f"\n前向传播测试:")
print(f"  Encoder 输入: {encoder_input.shape}")
print(f"  Decoder 输入: {decoder_input.shape}")

output = transformer.forward(encoder_input, decoder_input)
print(f"  输出形状: {output.shape}")
print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")

print("\n✅ Transformer 模型测试通过！")
```

---

## 🎯 实践练习

### 练习 1：可视化注意力权重

**任务**：训练一个小型 Transformer，可视化注意力权重。

### 练习 2：实现 Beam Search

**任务**：实现 Beam Search 解码策略。

---

## 📝 本章小结

### 核心要点

1. **Self-Attention**：让每个位置都能关注到所有位置
2. **Multi-Head**：并行计算多个注意力，捕获不同特征
3. **位置编码**：注入位置信息
4. **Encoder-Decoder**：编码器提取特征，解码器生成输出
5. **残差连接 + Layer Norm**：稳定训练

### 关键公式

```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
LayerNorm(x) = γ·(x-μ)/σ + β
```

### 下一步

下一章我们将学习预训练语言模型，理解 BERT 和 GPT 的原理。

---

<div align="center">

[⬅️ 上一章](../chapter07-attention/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter09-pretraining/README.md)

**🎯 Transformer 是大模型的基础，务必深入理解！**

</div>
