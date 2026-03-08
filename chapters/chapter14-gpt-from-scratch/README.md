# 第14章：从零实现 GPT

<div align="center">

[⬅️ 上一章](../chapter13-transformer-from-scratch/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter15-moe-from-scratch/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解 GPT 的 Decoder-only 架构
- ✅ 实现因果注意力掩码
- ✅ 训练一个简单的语言模型
- ✅ 实现文本生成功能
- ✅ 理解 GPT 的训练和推理流程

---

## 🎯 GPT 架构概述

### GPT vs Transformer

```
Transformer: Encoder + Decoder（机器翻译）
GPT: Decoder-only（文本生成）

GPT 特点:
1. 只有 Decoder 部分
2. 因果注意力掩码（只能看到之前的词）
3. 自回归生成
```

---

## 💻 完整 GPT 实现

```python
"""
mini-gpt/gpt.py - 从零实现 GPT
"""
import numpy as np

class GPT:
    """GPT 模型"""
    
    def __init__(self, 
                 vocab_size=50000,
                 n_layers=12,
                 n_heads=12,
                 d_model=768,
                 d_ff=3072,
                 max_seq_len=1024,
                 dropout=0.1):
        """
        初始化 GPT
        
        参数:
            vocab_size: 词汇表大小
            n_layers: 层数
            n_heads: 注意力头数
            d_model: 模型维度
            d_ff: 前馈网络维度
            max_seq_len: 最大序列长度
            dropout: Dropout 率
        """
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Token Embedding
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # Position Embedding（可学习）
        self.position_embedding = np.random.randn(max_seq_len, d_model) * 0.02
        
        # Transformer 层
        self.layers = [
            GPTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        
        # Layer Norm
        self.ln_f = LayerNorm(d_model)
        
        # 输出层
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
        
        self.dropout = dropout
    
    def create_causal_mask(self, seq_len):
        """创建因果掩码（上三角矩阵）"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask == 0  # 下三角为 True
    
    def forward(self, input_ids):
        """
        前向传播
        
        参数:
            input_ids: 输入 token IDs (batch_size, seq_len)
        
        返回:
            logits: 输出 logits (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token Embedding
        token_emb = self.token_embedding[input_ids]  # (batch, seq, d_model)
        
        # Position Embedding
        position_ids = np.arange(seq_len)
        pos_emb = self.position_embedding[position_ids]  # (seq, d_model)
        
        # 组合
        x = token_emb + pos_emb
        
        # Dropout
        if self.dropout > 0:
            mask = (np.random.rand(*x.shape) > self.dropout) / (1 - self.dropout)
            x = x * mask
        
        # 因果掩码
        causal_mask = self.create_causal_mask(seq_len)
        
        # Transformer 层
        for layer in self.layers:
            x = layer.forward(x, causal_mask)
        
        # Layer Norm
        x = self.ln_f.forward(x)
        
        # 输出层
        logits = np.matmul(x, self.lm_head)
        
        return logits
    
    def compute_loss(self, input_ids, target_ids):
        """计算损失"""
        logits = self.forward(input_ids)
        
        # Softmax
        probs = self.softmax(logits)
        
        # 交叉熵损失
        batch_size, seq_len = target_ids.shape
        loss = 0
        
        for b in range(batch_size):
            for t in range(seq_len):
                loss -= np.log(probs[b, t, target_ids[b, t]] + 1e-10)
        
        return loss / (batch_size * seq_len)
    
    def softmax(self, x):
        """Softmax"""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        文本生成
        
        参数:
            input_ids: 输入 token IDs (batch_size, seq_len)
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_k: Top-K 采样
        
        返回:
            生成的序列
        """
        for _ in range(max_new_tokens):
            # 截断到最大长度
            idx_cond = input_ids[:, -self.max_seq_len:]
            
            # 前向传播
            logits = self.forward(idx_cond)
            
            # 只取最后一个位置的 logits
            logits = logits[:, -1, :] / temperature
            
            # Top-K 采样
            if top_k is not None:
                v, _ = np.topk(logits, min(top_k, logits.size))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Softmax
            probs = self.softmax(logits)
            
            # 采样
            next_token = np.array([
                np.random.choice(self.vocab_size, p=probs[b])
                for b in range(input_ids.shape[0])
            ])
            
            # 拼接
            input_ids = np.concatenate([
                input_ids,
                next_token[:, np.newaxis]
            ], axis=1)
        
        return input_ids


class GPTBlock:
    """GPT 单层"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout):
        self.ln_1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.dropout = dropout
    
    def forward(self, x, mask=None):
        """前向传播"""
        # Self-Attention + 残差
        attn_output = self.attn.forward(self.ln_1.forward(x), mask)
        x = x + attn_output
        
        # MLP + 残差
        mlp_output = self.mlp.forward(self.ln_2.forward(x))
        x = x + mlp_output
        
        return x


class MLP:
    """前馈网络"""
    
    def __init__(self, d_model, d_ff, dropout):
        self.c_fc = np.random.randn(d_model, d_ff) * 0.02
        self.c_proj = np.random.randn(d_ff, d_model) * 0.02
        self.dropout = dropout
    
    def gelu(self, x):
        """GELU 激活函数"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def forward(self, x):
        """前向传播"""
        x = np.matmul(x, self.c_fc)
        x = self.gelu(x)
        x = np.matmul(x, self.c_proj)
        
        if self.dropout > 0:
            mask = (np.random.rand(*x.shape) > self.dropout) / (1 - self.dropout)
            x = x * mask
        
        return x


class MultiHeadAttention:
    """多头注意力"""
    
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.c_attn = np.random.randn(d_model, 3 * d_model) * 0.02
        self.c_proj = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, x, mask=None):
        """前向传播"""
        batch_size, seq_len, d_model = x.shape
        
        # QKV 投影
        qkv = np.matmul(x, self.c_attn)
        q, k, v = np.split(qkv, 3, axis=-1)
        
        # 分割成多头
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 注意力分数
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax
        attn_weights = self.softmax(scores)
        
        # 加权求和
        output = np.matmul(attn_weights, v)
        
        # 合并多头
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # 输出投影
        output = np.matmul(output, self.c_proj)
        
        return output
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


class LayerNorm:
    """层归一化"""
    
    def __init__(self, d_model, epsilon=1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.epsilon = epsilon
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.epsilon) + self.beta


# ================================
# 示例使用
# ================================

if __name__ == "__main__":
    print("初始化 GPT 模型...")
    
    # 创建模型（小型配置）
    gpt = GPT(
        vocab_size=1000,
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
        max_seq_len=128
    )
    
    print(f"✅ GPT 模型创建成功")
    print(f"   层数: {gpt.n_layers}")
    print(f"   模型维度: {gpt.d_model}")
    print(f"   注意力头数: {gpt.n_heads}")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 16
    
    input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\n前向传播测试:")
    print(f"   输入形状: {input_ids.shape}")
    
    logits = gpt.forward(input_ids)
    print(f"   输出形状: {logits.shape}")
    
    # 测试生成
    print(f"\n生成测试:")
    generated = gpt.generate(input_ids[:, :5], max_new_tokens=10)
    print(f"   生成序列形状: {generated.shape}")
    
    print("\n✅ GPT 模型测试通过！")
```

---

## 🎯 训练语言模型

```python
"""
mini-gpt/train.py - 训练脚本
"""
import numpy as np
from gpt import GPT

def train_gpt():
    """训练 GPT"""
    
    # 创建模型
    model = GPT(
        vocab_size=1000,
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
        max_seq_len=128
    )
    
    # 训练参数
    epochs = 100
    batch_size = 8
    seq_len = 32
    learning_rate = 3e-4
    
    print("开始训练 GPT...")
    
    for epoch in range(epochs):
        # 生成随机数据（实际应使用真实文本）
        input_ids = np.random.randint(1, 1000, (batch_size, seq_len))
        target_ids = np.random.randint(1, 1000, (batch_size, seq_len))
        
        # 计算损失
        loss = model.compute_loss(input_ids, target_ids)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
    
    print("✅ 训练完成！")
    
    # 测试生成
    print("\n生成文本:")
    start_tokens = np.array([[1, 2, 3]])  # 起始 token
    generated = model.generate(start_tokens, max_new_tokens=20, temperature=0.8)
    print(f"生成序列: {generated[0]}")

if __name__ == "__main__":
    train_gpt()
```

---

## 📝 本章小结

### 核心要点

1. **Decoder-only 架构**：GPT 只使用 Transformer 的 Decoder 部分
2. **因果掩码**：确保只能看到之前的词
3. **自回归生成**：逐词生成文本
4. **GELU 激活**：GPT 使用 GELU 而非 ReLU
5. **LayerNorm 位置**：GPT 在注意力之前做 LayerNorm

---

<div align="center">

[⬅️ 上一章](../chapter13-transformer-from-scratch/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter15-moe-from-scratch/README.md)

**🎉 恭喜！你已经从零实现了 GPT！**

</div>
