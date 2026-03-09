# 第16章：从零实现主流大模型

<div align="center">

[⬅️ 上一章](../chapter15-moe-from-scratch/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter17-peft/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 实现 LLaMA 架构（RoPE、RMSNorm、SwiGLU）
- ✅ 理解 Qwen 的中文优化
- ✅ 理解 DeepSeek 的 MoE 创新
- ✅ 加载和使用开源模型权重

---

## 🎯 LLaMA 架构实现

```python
"""
mini-llama/llama.py - LLaMA 架构实现
"""
import numpy as np

class LlamaModel:
    """LLaMA 模型"""
    
    def __init__(self,
                 vocab_size=32000,
                 n_layers=32,
                 n_heads=32,
                 d_model=4096,
                 d_ff=11008,
                 max_seq_len=2048,
                 rope_theta=10000.0):
        """
        初始化 LLaMA
        
        参数:
            vocab_size: 词汇表大小
            n_layers: 层数
            n_heads: 注意力头数
            d_model: 模型维度
            d_ff: 前馈网络维度
            max_seq_len: 最大序列长度
            rope_theta: RoPE theta
        """
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_model = d_model
        
        # Embedding
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # Transformer 层
        self.layers = [
            LlamaLayer(d_model, n_heads, d_ff, rope_theta)
            for _ in range(n_layers)
        ]
        
        # 输出
        self.rms_norm = RMSNorm(d_model)
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
    
    def forward(self, input_ids):
        """前向传播"""
        # Embedding
        x = self.token_embedding[input_ids]
        
        # 层
        for layer in self.layers:
            x = layer.forward(x)
        
        # 输出
        x = self.rms_norm.forward(x)
        logits = np.matmul(x, self.lm_head)
        
        return logits


class LlamaLayer:
    """LLaMA 层"""
    
    def __init__(self, d_model, n_heads, d_ff, rope_theta):
        self.attention = LlamaAttention(d_model, n_heads, rope_theta)
        self.feed_forward = LlamaMLP(d_model, d_ff)
        self.input_layernorm = RMSNorm(d_model)
        self.post_attention_layernorm = RMSNorm(d_model)
    
    def forward(self, x):
        # Self-Attention
        residual = x
        x = self.input_layernorm.forward(x)
        x = self.attention.forward(x)
        x = residual + x
        
        # MLP
        residual = x
        x = self.post_attention_layernorm.forward(x)
        x = self.feed_forward.forward(x)
        x = residual + x
        
        return x


class LlamaAttention:
    """LLaMA 注意力（带 RoPE）"""
    
    def __init__(self, d_model, n_heads, rope_theta):
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.rope_theta = rope_theta
        
        # QKV
        self.q_proj = np.random.randn(d_model, d_model) * 0.02
        self.k_proj = np.random.randn(d_model, d_model) * 0.02
        self.v_proj = np.random.randn(d_model, d_model) * 0.02
        self.o_proj = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # QKV
        Q = np.matmul(x, self.q_proj)
        K = np.matmul(x, self.k_proj)
        V = np.matmul(x, self.v_proj)
        
        # 应用 RoPE
        Q = self.apply_rotary_emb(Q)
        K = self.apply_rotary_emb(K)
        
        # 分割多头
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 注意力
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # 因果掩码
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        scores = np.where(mask == 1, -1e9, scores)
        
        attn_weights = self.softmax(scores)
        output = np.matmul(attn_weights, V)
        
        # 合并
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        return np.matmul(output, self.o_proj)
    
    def apply_rotary_emb(self, x):
        """应用旋转位置编码"""
        seq_len = x.shape[1]
        d = x.shape[-1]
        
        # 生成频率
        freqs = 1.0 / (self.rope_theta ** (np.arange(0, d, 2, dtype=np.float32) / d))
        
        # 生成位置
        positions = np.arange(seq_len)
        
        # 计算
        freqs = np.outer(positions, freqs)
        
        # 旋转
        cos = np.cos(freqs)
        sin = np.sin(freqs)
        
        # 应用
        x_rot = np.zeros_like(x)
        x_rot[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        x_rot[..., 1::2] = x[..., 1::2] * cos + x[..., 0::2] * sin
        
        return x_rot
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


class LlamaMLP:
    """LLaMA MLP（SwiGLU）"""
    
    def __init__(self, d_model, d_ff):
        self.gate_proj = np.random.randn(d_model, d_ff) * 0.02
        self.up_proj = np.random.randn(d_model, d_ff) * 0.02
        self.down_proj = np.random.randn(d_ff, d_model) * 0.02
    
    def silu(self, x):
        """SiLU 激活函数"""
        return x * (1 / (1 + np.exp(-x)))
    
    def forward(self, x):
        # SwiGLU: down(silu(gate) * up)
        gate = np.matmul(x, self.gate_proj)
        up = np.matmul(x, self.up_proj)
        
        # SiLU + 门控
        hidden = self.silu(gate) * up
        
        # 下投影
        output = np.matmul(hidden, self.down_proj)
        
        return output


class RMSNorm:
    """RMS 归一化"""
    
    def __init__(self, d_model, epsilon=1e-6):
        self.weight = np.ones(d_model)
        self.epsilon = epsilon
    
    def forward(self, x):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)
        return self.weight * x / rms


# ================================
# 示例
# ================================

if __name__ == "__main__":
    print("初始化 LLaMA 模型...")
    
    model = LlamaModel(
        vocab_size=1000,
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=512,
        max_seq_len=128
    )
    
    print(f"✅ LLaMA 模型创建成功")
    print(f"   关键特性: RoPE、RMSNorm、SwiGLU")
    
    # 测试
    input_ids = np.random.randint(0, 1000, (2, 16))
    logits = model.forward(input_ids)
    
    print(f"\n前向传播测试:")
    print(f"   输入: {input_ids.shape}")
    print(f"   输出: {logits.shape}")
    
    print("\n✅ LLaMA 模型测试通过！")
```

---

## 📝 本章小结

### 核心架构对比

| 模型 | 特点 | 关键技术 |
|------|------|----------|
| **LLaMA** | 开源基座模型 | RoPE、RMSNorm、SwiGLU |
| **Qwen** | 中文优化 | 长文本、多语言 |
| **DeepSeek** | MoE 架构 | DeepSeek-MoE、高效率 |
| **Mistral** | 高性能小模型 | Sliding Window、GQA |

---

<div align="center">

[⬅️ 上一章](../chapter15-moe-from-scratch/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter17-peft/README.md)

</div>
