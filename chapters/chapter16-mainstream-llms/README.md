# 第16章：主流大模型架构解析

<div align="center">

[⬅️ 上一章](../chapter15-moe-from-scratch/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter17-peft/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解 LLaMA 系列架构特点
- ✅ 掌握 RoPE、RMSNorm、SwiGLU 等核心组件
- ✅ 了解 Qwen、DeepSeek、Mistral 等模型创新
- ✅ 理解模型参数规模与性能的关系

---

## 🎯 主流开源模型概览

### 开源模型发展时间线

```
2023年:
├── 02月 - LLaMA-1 发布（Meta）
├── 07月 - LLaMA-2 发布（商业许可）
├── 08月 - Qwen-7B 发布（阿里）
├── 09月 - Mistral-7B 发布
├── 11月 - DeepSeek-67B 发布
└── 12月 - Mixtral-8x7B 发布（MoE）

2024年:
├── 02月 - Gemma 发布（Google）
├── 04月 - LLaMA-3 发布
├── 05月 - Qwen2 发布
├── 12月 - DeepSeek-V3 发布
└── ...
```

### 模型参数对比

| 模型 | 参数量 | 架构特点 | 训练数据 |
|------|--------|----------|----------|
| LLaMA-2-7B | 7B | 标准 Transformer | 2T tokens |
| LLaMA-2-70B | 70B | GQA | 2T tokens |
| Qwen2-7B | 7B | 长文本优化 | ~7T tokens |
| Mistral-7B | 7B | Sliding Window + GQA | 未知 |
| Mixtral-8x7B | 47B (13B激活) | MoE | 未知 |
| DeepSeek-V3 | 671B (37B激活) | MoE | 14.8T tokens |

---

## 1. LLaMA 架构详解

### 1.1 架构特点

```
LLaMA 核心创新：

1. RoPE（旋转位置编码）
   - 相对位置编码
   - 更好的长度外推能力

2. RMSNorm（均方根归一化）
   - 比 LayerNorm 更简单高效
   - 去掉均值计算

3. SwiGLU（激活函数）
   - 门控线性单元
   - 比 ReLU、GELU 效果更好

4. GQA（分组查询注意力）
   - 减少 KV Cache 大小
   - 加速推理（LLaMA-2 70B）
```

### 1.2 RoPE 实现

```python
import numpy as np

class RotaryPositionalEmbedding:
    """旋转位置编码"""
    
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
        self.inv_freq = inv_freq
        
        # 预计算 cos 和 sin
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        """构建缓存"""
        t = np.arange(seq_len)
        freqs = np.outer(t, self.inv_freq)
        
        # 拼接形成完整旋转角度
        emb = np.concatenate([freqs, freqs], axis=-1)
        
        self.cos_cache = np.cos(emb)
        self.sin_cache = np.sin(emb)
    
    def forward(self, x, positions=None):
        """
        应用 RoPE
        
        参数:
            x: 输入张量 (batch, seq_len, n_heads, head_dim)
            positions: 位置索引（可选）
        """
        seq_len = x.shape[1]
        
        if positions is None:
            cos = self.cos_cache[:seq_len]
            sin = self.sin_cache[:seq_len]
        else:
            cos = self.cos_cache[positions]
            sin = self.sin_cache[positions]
        
        # 扩展维度以匹配
        cos = cos[np.newaxis, :, np.newaxis, :]  # (1, seq, 1, dim)
        sin = sin[np.newaxis, :, np.newaxis, :]
        
        # 旋转
        x_rot = self._rotate_half(x)
        return x * cos + x_rot * sin
    
    def _rotate_half(self, x):
        """旋转一半维度"""
        d = x.shape[-1] // 2
        return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


# 示例
rope = RotaryPositionalEmbedding(dim=64)
x = np.random.randn(2, 10, 8, 64)  # (batch, seq, heads, dim)
output = rope.forward(x)
print(f"RoPE 输出形状: {output.shape}")
```

### 1.3 RMSNorm 实现

```python
import numpy as np

class RMSNorm:
    """均方根归一化"""
    
    def __init__(self, dim, epsilon=1e-6):
        self.dim = dim
        self.epsilon = epsilon
        self.weight = np.ones(dim)
    
    def forward(self, x):
        """
        前向传播
        
        RMSNorm vs LayerNorm:
        - LayerNorm: (x - mean) / std
        - RMSNorm: x / sqrt(mean(x^2))
        """
        # 计算 RMS
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)
        
        # 归一化并缩放
        return self.weight * (x / rms)
    
    def __call__(self, x):
        return self.forward(x)


# 对比 LayerNorm
class LayerNorm:
    """层归一化（对比用）"""
    
    def __init__(self, dim, epsilon=1e-6):
        self.dim = dim
        self.epsilon = epsilon
        self.weight = np.ones(dim)
        self.bias = np.zeros(dim)
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        return self.weight * x_norm + self.bias


# 性能对比
x = np.random.randn(100, 768)

rms_norm = RMSNorm(768)
layer_norm = LayerNorm(768)

import time

# RMSNorm
start = time.time()
for _ in range(10000):
    _ = rms_norm(x)
rms_time = time.time() - start

# LayerNorm
start = time.time()
for _ in range(10000):
    _ = layer_norm(x)
ln_time = time.time() - start

print(f"RMSNorm: {rms_time:.4f}s")
print(f"LayerNorm: {ln_time:.4f}s")
print(f"加速比: {ln_time/rms_time:.2f}x")
```

### 1.4 SwiGLU 实现

```python
import numpy as np

class SwiGLU:
    """SwiGLU 激活函数"""
    
    def __init__(self, d_model, d_ff):
        """
        参数:
            d_model: 输入维度
            d_ff: 中间维度（通常为 d_model * 8/3）
        """
        # 三个线性层
        self.gate_proj = np.random.randn(d_model, d_ff) * 0.02
        self.up_proj = np.random.randn(d_model, d_ff) * 0.02
        self.down_proj = np.random.randn(d_ff, d_model) * 0.02
    
    def silu(self, x):
        """SiLU/Swish 激活函数"""
        return x * (1 / (1 + np.exp(-x)))
    
    def forward(self, x):
        """
        SwiGLU: down(silu(gate(x)) * up(x))
        
        对比标准 FFN:
        - 标准: W2 * activation(W1 * x)
        - SwiGLU: W3 * silu(W1 * x) * W2 * x
        """
        # 门控路径
        gate = np.matmul(x, self.gate_proj)
        gate = self.silu(gate)
        
        # 上投影路径
        up = np.matmul(x, self.up_proj)
        
        # 门控相乘
        hidden = gate * up
        
        # 下投影
        output = np.matmul(hidden, self.down_proj)
        
        return output


# 对比不同激活函数
class StandardFFN:
    """标准 FFN（GELU）"""
    
    def __init__(self, d_model, d_ff):
        self.up_proj = np.random.randn(d_model, d_ff) * 0.02
        self.down_proj = np.random.randn(d_ff, d_model) * 0.02
    
    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def forward(self, x):
        return np.matmul(self.gelu(np.matmul(x, self.up_proj)), self.down_proj)
```

### 1.5 GQA（分组查询注意力）

```python
import numpy as np

class GroupedQueryAttention:
    """分组查询注意力"""
    
    def __init__(self, d_model, n_heads, n_kv_heads):
        """
        参数:
            d_model: 模型维度
            n_heads: 查询头数
            n_kv_heads: KV 头数（较少）
        """
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # 每个 KV 头对应的 Q 头数
        self.d_k = d_model // n_heads
        
        self.q_proj = np.random.randn(d_model, d_model) * 0.02
        self.k_proj = np.random.randn(d_model, d_model // self.n_rep) * 0.02
        self.v_proj = np.random.randn(d_model, d_model // self.n_rep) * 0.02
        self.o_proj = np.random.randn(d_model, d_model) * 0.02
    
    def repeat_kv(self, x):
        """重复 KV 头以匹配 Q 头数"""
        # x: (batch, n_kv_heads, seq, d_k)
        batch, n_kv, seq, d_k = x.shape
        
        # 重复
        x = x[:, :, np.newaxis, :, :]  # (batch, n_kv, 1, seq, d_k)
        x = np.tile(x, (1, 1, self.n_rep, 1, 1))  # (batch, n_kv, n_rep, seq, d_k)
        x = x.reshape(batch, n_kv * self.n_rep, seq, d_k)  # (batch, n_heads, seq, d_k)
        
        return x
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # QKV 投影
        Q = np.matmul(x, self.q_proj)
        K = np.matmul(x, self.k_proj)
        V = np.matmul(x, self.v_proj)
        
        # 分头
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 重复 KV
        K = self.repeat_kv(K)
        V = self.repeat_kv(V)
        
        # 注意力计算
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # 因果掩码
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        scores = np.where(mask == 1, -1e9, scores)
        
        # Softmax
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        # 输出
        output = np.matmul(attn_weights, V)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        return np.matmul(output, self.o_proj)


# 对比 MHA、MQA、GQA
print("注意力机制对比:")
print("MHA (Multi-Head Attention): 所有头独立 K、V")
print("MQA (Multi-Query Attention): 所有头共享 K、V")
print("GQA (Grouped-Query Attention): 组内共享 K、V（折中方案）")
print("\nKV Cache 大小对比 (n_heads=32, seq_len=4096, d_k=128):")
print("MHA: 32 * 4096 * 128 * 2 = 134M 元素")
print("MQA: 1 * 4096 * 128 * 2 = 4M 元素")
print("GQA (8组): 8 * 4096 * 128 * 2 = 16M 元素")
```

---

## 2. Qwen 架构特点

### 2.1 中文优化

```
Qwen 系列特点：

1. 中文词表优化
   - 中文分词更高效
   - 压缩率更高

2. 长文本支持
   - Qwen2 支持 32K 上下文
   - 使用 Yarn 或 NTK-aware 插值

3. 多语言能力
   - 支持多种语言
   - 代码能力强

4. 模型家族
   - Qwen-0.5B ~ Qwen-72B
   - Qwen-VL（多模态）
   - Qwen-Math（数学）
```

### 2.2 Qwen 模型配置

```python
# Qwen-7B 配置示例
qwen_config = {
    "vocab_size": 152064,      # 扩展词表（支持中文）
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 32,          # 标准 MHA
    "d_model": 4096,
    "d_ff": 13696,             # FFN 中间维度
    "max_seq_len": 32768,      # 长文本支持
    "rope_theta": 10000.0,
    "rope_scaling": {
        "type": "yarn",
        "factor": 4.0          # 长度扩展因子
    }
}

# Qwen2-7B 配置
qwen2_config = {
    "vocab_size": 152064,
    "n_layers": 28,
    "n_heads": 28,
    "n_kv_heads": 4,           # 使用 GQA！
    "d_model": 3584,
    "d_ff": 18944,
    "max_seq_len": 32768,
    "rope_theta": 1000000.0,   # 更大的 theta
}
```

---

## 3. DeepSeek 架构创新

### 3.1 DeepSeek-MoE

```
DeepSeek MoE 创新：

1. 细粒度专家分割
   - 将专家分成更小的单元
   - 更灵活的路由

2. 共享专家
   - 部分专家始终激活
   - 捕获通用知识

3. 负载均衡
   - 专家级平衡损失
   - 设备级平衡损失
```

### 3.2 DeepSeek-V3 特点

```python
# DeepSeek-V3 配置
deepseek_v3_config = {
    "vocab_size": 102400,
    "n_layers": 61,
    "n_heads": 128,
    "n_kv_heads": 128,
    "d_model": 7168,
    "d_ff": 18432,
    
    # MoE 配置
    "n_routed_experts": 256,    # 路由专家数
    "n_shared_experts": 1,      # 共享专家数
    "n_activated_experts": 8,   # 每次激活专家数
    
    # 多头潜在注意力（MLA）
    "use_mla": True,
    
    # 参数
    "total_params": 671_000_000_000,     # 671B
    "active_params": 37_000_000_000,     # 37B 激活
}
```

---

## 4. Mistral 架构特点

### 4.1 Sliding Window Attention

```python
import numpy as np

class SlidingWindowAttention:
    """滑动窗口注意力"""
    
    def __init__(self, d_model, n_heads, window_size=4096):
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.d_k = d_model // n_heads
        
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
        
        # 分头
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 注意力分数
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # 滑动窗口掩码
        mask = np.ones((seq_len, seq_len)) * -1e9
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i+1] = 0
        
        scores = scores + mask
        
        # Softmax
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        # 输出
        output = np.matmul(attn_weights, V)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        return np.matmul(output, self.o_proj)


print("滑动窗口注意力优势:")
print(f"- 降低复杂度: O(n * window_size) vs O(n^2)")
print(f"- 固定 KV Cache 大小")
print(f"- 适合长文本")
```

---

## 5. 模型参数量计算

### 5.1 标准 Transformer 参数量

```python
def calc_transformer_params(vocab_size, n_layers, d_model, d_ff, n_heads):
    """计算 Transformer 参数量"""
    
    # Embedding
    embed_params = vocab_size * d_model
    
    # 单层参数
    # Self-Attention: QKV + O
    attn_params = 4 * d_model * d_model
    
    # FFN: up + down + gate (SwiGLU)
    ffn_params = 3 * d_model * d_ff
    
    # LayerNorm: 2 * d_model (每层两个)
    ln_params = 2 * 2 * d_model
    
    # 单层总计
    layer_params = attn_params + ffn_params + ln_params
    
    # 总参数
    total_params = embed_params + n_layers * layer_params + d_model * vocab_size  # lm_head
    
    return {
        "embedding": embed_params,
        "per_layer": layer_params,
        "total": total_params,
        "total_B": total_params / 1e9
    }


# 计算常见模型
models = [
    ("GPT-2 Small", 50257, 12, 768, 3072, 12),
    ("GPT-2 Medium", 50257, 24, 1024, 4096, 16),
    ("LLaMA-7B", 32000, 32, 4096, 11008, 32),
    ("LLaMA-13B", 32000, 40, 5120, 13824, 40),
    ("LLaMA-70B", 32000, 80, 8192, 28672, 64),
]

print("模型参数量估算:")
print("-" * 60)
for name, vocab, layers, d_model, d_ff, n_heads in models:
    params = calc_transformer_params(vocab, layers, d_model, d_ff, n_heads)
    print(f"{name:15} {params['total_B']:.2f}B 参数")
```

---

## 💻 完整 LLaMA 模型实现

```python
"""
完整 LLaMA 模型实现（简化版）
"""
import numpy as np

class LlamaModel:
    """LLaMA 模型"""
    
    def __init__(self,
                 vocab_size=32000,
                 n_layers=32,
                 n_heads=32,
                 n_kv_heads=None,  # GQA
                 d_model=4096,
                 d_ff=11008,
                 max_seq_len=2048,
                 rope_theta=10000.0):
        
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        
        # Embedding
        self.embed_tokens = np.random.randn(vocab_size, d_model) * 0.02
        
        # Transformer 层
        self.layers = [
            LlamaDecoderLayer(d_model, n_heads, self.n_kv_heads, d_ff, rope_theta)
            for _ in range(n_layers)
        ]
        
        # 输出
        self.norm = RMSNorm(d_model)
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
    
    def forward(self, input_ids):
        """前向传播"""
        # Embedding
        x = self.embed_tokens[input_ids]
        
        # Transformer 层
        for layer in self.layers:
            x = layer.forward(x)
        
        # 输出
        x = self.norm.forward(x)
        logits = np.matmul(x, self.lm_head)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """文本生成"""
        for _ in range(max_new_tokens):
            # 前向传播
            logits = self.forward(input_ids)
            
            # 取最后一个位置的 logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # 采样
            probs = np.exp(next_token_logits - np.max(next_token_logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            next_token = np.array([np.random.choice(len(probs[0]), p=probs[0])])
            
            # 拼接
            input_ids = np.concatenate([input_ids, next_token[np.newaxis, :]], axis=1)
        
        return input_ids


class LlamaDecoderLayer:
    """LLaMA 解码层"""
    
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, rope_theta):
        self.self_attn = LlamaAttention(d_model, n_heads, n_kv_heads, rope_theta)
        self.mlp = LlamaMLP(d_model, d_ff)
        self.input_layernorm = RMSNorm(d_model)
        self.post_attention_layernorm = RMSNorm(d_model)
    
    def forward(self, x):
        # Self-Attention
        residual = x
        x = self.input_layernorm.forward(x)
        x = self.self_attn.forward(x)
        x = residual + x
        
        # MLP
        residual = x
        x = self.post_attention_layernorm.forward(x)
        x = self.mlp.forward(x)
        x = residual + x
        
        return x


class LlamaAttention:
    """LLaMA 注意力"""
    
    def __init__(self, d_model, n_heads, n_kv_heads, rope_theta):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        self.rope_theta = rope_theta
        
        head_dim = d_model // n_heads
        self.q_proj = np.random.randn(d_model, n_heads * head_dim) * 0.02
        self.k_proj = np.random.randn(d_model, n_kv_heads * head_dim) * 0.02
        self.v_proj = np.random.randn(d_model, n_kv_heads * head_dim) * 0.02
        self.o_proj = np.random.randn(n_heads * head_dim, d_model) * 0.02
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(head_dim, base=rope_theta)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # QKV
        Q = np.matmul(x, self.q_proj)
        K = np.matmul(x, self.k_proj)
        V = np.matmul(x, self.v_proj)
        
        # 分头
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # RoPE
        Q = self.rope.forward(Q)
        K = self.rope.forward(K)
        
        # GQA: 重复 KV
        if self.n_rep > 1:
            K = np.repeat(K, self.n_rep, axis=1)
            V = np.repeat(V, self.n_rep, axis=1)
        
        # 注意力
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # 因果掩码
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        scores = np.where(mask == 1, -1e9, scores)
        
        # Softmax
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        # 输出
        output = np.matmul(attn_weights, V)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        return np.matmul(output, self.o_proj)


class LlamaMLP:
    """LLaMA MLP"""
    
    def __init__(self, d_model, d_ff):
        self.gate_proj = np.random.randn(d_model, d_ff) * 0.02
        self.up_proj = np.random.randn(d_model, d_ff) * 0.02
        self.down_proj = np.random.randn(d_ff, d_model) * 0.02
    
    def silu(self, x):
        return x * (1 / (1 + np.exp(-x)))
    
    def forward(self, x):
        return np.matmul(self.silu(np.matmul(x, self.gate_proj)) * np.matmul(x, self.up_proj), 
                        self.down_proj)


class RMSNorm:
    """RMS 归一化"""
    
    def __init__(self, dim, epsilon=1e-6):
        self.weight = np.ones(dim)
        self.epsilon = epsilon
    
    def forward(self, x):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)
        return self.weight * x / rms


class RotaryPositionalEmbedding:
    """旋转位置编码"""
    
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
        t = np.arange(max_seq_len)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        self.cos_cache = np.cos(emb)
        self.sin_cache = np.sin(emb)
    
    def forward(self, x):
        seq_len = x.shape[2]
        cos = self.cos_cache[:seq_len][np.newaxis, np.newaxis, :, :]
        sin = self.sin_cache[:seq_len][np.newaxis, np.newaxis, :, :]
        
        x_rot = np.concatenate([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], axis=-1)
        return x * cos + x_rot * sin


# 测试
if __name__ == "__main__":
    print("创建 LLaMA 模型...")
    
    model = LlamaModel(
        vocab_size=1000,
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=512,
        max_seq_len=128
    )
    
    print("✅ LLaMA 模型创建成功")
    
    # 测试前向传播
    input_ids = np.random.randint(0, 1000, (1, 16))
    logits = model.forward(input_ids)
    print(f"输入: {input_ids.shape}, 输出: {logits.shape}")
    
    print("\n✅ 测试通过!")
```

---

## 📝 本章小结

### 核心要点

1. **LLaMA 系列**：RoPE + RMSNorm + SwiGLU + GQA 的组合
2. **Qwen 系列**：中文词表优化 + 长文本支持
3. **DeepSeek 系列**：MoE 创新架构，高效参数利用
4. **Mistral**：滑动窗口注意力 + GQA

### 架构选择指南

| 需求 | 推荐架构 | 原因 |
|------|----------|------|
| 高效推理 | GQA + MQA | 减少 KV Cache |
| 长文本 | RoPE + 滑动窗口 | 更好长度外推 |
| 大规模 | MoE | 参数效率高 |
| 中文任务 | Qwen 系列 | 词表优化 |

---

<div align="center">

[⬅️ 上一章](../chapter15-moe-from-scratch/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter17-peft/README.md)]

</div>
