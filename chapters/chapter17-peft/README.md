# 第17章：参数高效微调（PEFT）

<div align="center">

[⬅️ 上一章](../chapter16-mainstream-llms/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter18-instruction-tuning-rlhf/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解 PEFT 的原理和优势
- ✅ 从零实现 LoRA 算法
- ✅ 掌握 Prefix Tuning、Adapter 等方法
- ✅ 使用 Hugging Face PEFT 库进行微调

---

## 🎯 为什么需要 PEFT？

### 全参数微调的问题

```
全参数微调（Full Fine-tuning）的问题：

1. 显存需求大
   - LLaMA-7B: 需要 ~28GB 显存（FP16）
   - LLaMA-70B: 需要 ~280GB 显存

2. 存储成本高
   - 每个任务一份完整权重
   - 100 个任务 = 100 份权重

3. 部署困难
   - 每个任务需要独立模型
   - 切换成本高

PEFT 解决方案：
- 只训练少量参数（0.1%~1%）
- 保持预训练权重冻结
- 多任务共享基础模型
```

### PEFT 方法概览

```
PEFT 方法分类：

┌─────────────────────────────────────────────┐
│                 PEFT 方法                    │
├──────────────┬──────────────┬───────────────┤
│   增量式      │   适配器式    │   重参数化式   │
├──────────────┼──────────────┼───────────────┤
│ Prefix Tuning│   Adapter    │     LoRA      │
│ Prompt Tuning│  Adapters    │    AdaLoRA    │
│  Soft Prompt │   IA³        │    QLoRA      │
└──────────────┴──────────────┴───────────────┘
```

---

## 1. LoRA（Low-Rank Adaptation）

### 1.1 LoRA 原理

```
LoRA 核心思想：

预训练权重 W ∈ R^{d×k} 冻结

添加低秩分解：
  W' = W + ΔW = W + B·A
  
其中：
  B ∈ R^{d×r}  (r << min(d,k))
  A ∈ R^{r×k}
  
参数量：
  原始: d × k
  LoRA: d × r + r × k = r(d + k)
  
当 r << d,k 时，参数量大幅减少
```

### 1.2 LoRA 实现

```python
import numpy as np

class LoRALayer:
    """LoRA 层"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0):
        """
        初始化 LoRA
        
        参数:
            in_features: 输入维度
            out_features: 输出维度
            rank: LoRA 秩
            alpha: 缩放因子
            dropout: Dropout 率
        """
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = dropout
        
        # LoRA 权重
        # A 用随机初始化，B 初始化为零
        # 这样初始时 ΔW = B·A = 0，不改变原模型
        self.lora_A = np.random.randn(in_features, rank) * 0.01
        self.lora_B = np.zeros((rank, out_features))
        
        # 预训练权重（冻结）
        self.W = None
        
    def set_pretrained_weight(self, W):
        """设置预训练权重"""
        self.W = W
    
    def forward(self, x):
        """
        前向传播
        
        公式: output = W·x + (alpha/r)·B·A·x
        """
        # 原始输出（冻结）
        original = np.matmul(x, self.W) if self.W is not None else 0
        
        # LoRA 输出
        lora = np.matmul(x, np.matmul(self.lora_A, self.lora_B))
        
        # 缩放
        output = original + self.scaling * lora
        
        return output
    
    def get_delta_weight(self):
        """获取 LoRA 增量权重"""
        return self.scaling * np.matmul(self.lora_B, self.lora_A.T)
    
    def merge_weights(self):
        """合并 LoRA 权重到原模型"""
        if self.W is not None:
            self.W = self.W + self.get_delta_weight()
            # 清零 LoRA 权重
            self.lora_A = np.zeros_like(self.lora_A)
            self.lora_B = np.zeros_like(self.lora_B)


class LoRALinear:
    """带 LoRA 的线性层"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        self.in_features = in_features
        self.out_features = out_features
        
        # 原始线性层权重
        self.weight = np.random.randn(in_features, out_features) * 0.02
        self.bias = np.zeros(out_features)
        
        # LoRA
        self.lora = LoRALayer(in_features, out_features, rank, alpha)
        self.lora.set_pretrained_weight(self.weight)
    
    def forward(self, x):
        """前向传播"""
        # LoRA 输出
        lora_out = self.lora.forward(x)
        
        # 加偏置
        return lora_out + self.bias
    
    def train_lora_only(self):
        """只训练 LoRA 参数"""
        # 冻结原权重
        self.weight.flags.writeable = False


# 参数量对比
def compare_params():
    """对比全参数和 LoRA 参数量"""
    d = 4096  # LLaMA-7B hidden size
    
    # 全参数
    full_params = d * d
    
    # 不同 rank 的 LoRA
    for rank in [4, 8, 16, 32, 64]:
        lora_params = 2 * d * rank  # A + B
        ratio = lora_params / full_params * 100
        print(f"rank={rank:2d}: {lora_params:>10,} 参数 ({ratio:.2f}%)")
    
    print(f"全参数:   {full_params:>10,} 参数 (100%)")

compare_params()
```

### 1.3 完整 LoRA 模型

```python
import numpy as np

class LoRAConfig:
    """LoRA 配置"""
    
    def __init__(self,
                 r=8,
                 lora_alpha=16,
                 lora_dropout=0.0,
                 target_modules=["q_proj", "v_proj"],
                 bias="none"):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.bias = bias


class LoraSelfAttention:
    """带 LoRA 的自注意力层"""
    
    def __init__(self, d_model, n_heads, lora_config):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 原始投影层
        self.q_proj = np.random.randn(d_model, d_model) * 0.02
        self.k_proj = np.random.randn(d_model, d_model) * 0.02
        self.v_proj = np.random.randn(d_model, d_model) * 0.02
        self.o_proj = np.random.randn(d_model, d_model) * 0.02
        
        # 添加 LoRA
        self.lora_q = None
        self.lora_v = None
        
        if "q_proj" in lora_config.target_modules:
            self.lora_q = LoRALayer(d_model, d_model, 
                                    lora_config.r, 
                                    lora_config.lora_alpha)
            self.lora_q.set_pretrained_weight(self.q_proj)
        
        if "v_proj" in lora_config.target_modules:
            self.lora_v = LoRALayer(d_model, d_model,
                                    lora_config.r,
                                    lora_config.lora_alpha)
            self.lora_v.set_pretrained_weight(self.v_proj)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Q 投影（可能带 LoRA）
        if self.lora_q is not None:
            Q = self.lora_q.forward(x)
        else:
            Q = np.matmul(x, self.q_proj)
        
        # K 投影
        K = np.matmul(x, self.k_proj)
        
        # V 投影（可能带 LoRA）
        if self.lora_v is not None:
            V = self.lora_v.forward(x)
        else:
            V = np.matmul(x, self.v_proj)
        
        # 分头
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 注意力
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        # 输出
        output = np.matmul(attn_weights, V)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        return np.matmul(output, self.o_proj)


class LoraMLP:
    """带 LoRA 的 MLP 层"""
    
    def __init__(self, d_model, d_ff, lora_config):
        self.gate_proj = np.random.randn(d_model, d_ff) * 0.02
        self.up_proj = np.random.randn(d_model, d_ff) * 0.02
        self.down_proj = np.random.randn(d_ff, d_model) * 0.02
        
        # 可选：为 MLP 添加 LoRA
        self.lora_gate = None
        self.lora_up = None
        self.lora_down = None
        
        if "gate_proj" in lora_config.target_modules:
            self.lora_gate = LoRALayer(d_model, d_ff,
                                       lora_config.r,
                                       lora_config.lora_alpha)
            self.lora_gate.set_pretrained_weight(self.gate_proj)
    
    def silu(self, x):
        return x * (1 / (1 + np.exp(-x)))
    
    def forward(self, x):
        if self.lora_gate is not None:
            gate = self.lora_gate.forward(x)
        else:
            gate = np.matmul(x, self.gate_proj)
        
        up = np.matmul(x, self.up_proj)
        
        return np.matmul(self.silu(gate) * up, self.down_proj)
```

---

## 2. 其他 PEFT 方法

### 2.1 Prefix Tuning

```python
import numpy as np

class PrefixTuning:
    """Prefix Tuning"""
    
    def __init__(self, num_layers, d_model, prefix_length=10, num_heads=8):
        """
        参数:
            num_layers: 层数
            d_model: 模型维度
            prefix_length: 前缀长度
            num_heads: 注意力头数
        """
        self.prefix_length = prefix_length
        self.num_layers = num_layers
        
        # 每层的前缀向量（可学习）
        # 对于注意力层，需要为每个头生成 K 和 V 的前缀
        self.prefix_keys = [
            np.random.randn(prefix_length, num_heads, d_model // num_heads) * 0.02
            for _ in range(num_layers)
        ]
        self.prefix_values = [
            np.random.randn(prefix_length, num_heads, d_model // num_heads) * 0.02
            for _ in range(num_layers)
        ]
        
        # 或者使用更小的 MLP 来生成前缀
        self.prefix_embedding = np.random.randn(prefix_length, d_model) * 0.02
        self.mlp = None  # 可选的 MLP
    
    def get_prefix_kv(self, layer_idx):
        """获取某一层的前缀 K 和 V"""
        return self.prefix_keys[layer_idx], self.prefix_values[layer_idx]
    
    def forward_with_prefix(self, x, layer_idx):
        """
        在注意力计算中加入前缀
        
        原始: attention(x, K, V)
        带前缀: attention(x, [prefix_K; K], [prefix_V; V])
        """
        prefix_k, prefix_v = self.get_prefix_kv(layer_idx)
        
        # prefix_k: (prefix_len, num_heads, d_k)
        # prefix_v: (prefix_len, num_heads, d_k)
        
        return prefix_k, prefix_v


class PrefixTuningAttention:
    """带 Prefix Tuning 的注意力层"""
    
    def __init__(self, d_model, n_heads, prefix_length=10):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.prefix_length = prefix_length
        
        # 原始投影（冻结）
        self.q_proj = np.random.randn(d_model, d_model) * 0.02
        self.k_proj = np.random.randn(d_model, d_model) * 0.02
        self.v_proj = np.random.randn(d_model, d_model) * 0.02
        self.o_proj = np.random.randn(d_model, d_model) * 0.02
        
        # 前缀参数（可训练）
        self.prefix_k = np.random.randn(prefix_length, n_heads, self.d_k) * 0.02
        self.prefix_v = np.random.randn(prefix_length, n_heads, self.d_k) * 0.02
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V 投影
        Q = np.matmul(x, self.q_proj)
        K = np.matmul(x, self.k_proj)
        V = np.matmul(x, self.v_proj)
        
        # 分头
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 添加前缀
        # prefix_k: (prefix_len, num_heads, d_k) -> (1, num_heads, prefix_len, d_k)
        prefix_k = self.prefix_k.transpose(1, 0, 2)[np.newaxis, :, :, :]
        prefix_v = self.prefix_v.transpose(1, 0, 2)[np.newaxis, :, :, :]
        
        # 拼接: (batch, heads, prefix_len + seq_len, d_k)
        K = np.concatenate([np.tile(prefix_k, (batch_size, 1, 1, 1)), K], axis=2)
        V = np.concatenate([np.tile(prefix_v, (batch_size, 1, 1, 1)), V], axis=2)
        
        # 注意力计算
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # 注意：因果掩码需要调整
        # 前缀部分可以看到所有内容
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        # 输出
        output = np.matmul(attn_weights, V)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        return np.matmul(output, self.o_proj)


# 参数量对比
def prefix_params_comparison():
    d_model = 4096
    num_layers = 32
    prefix_length = 10
    
    # 全参数微调
    full_params = num_layers * 4 * d_model * d_model  # 4个投影层
    
    # Prefix Tuning
    prefix_params = num_layers * 2 * prefix_length * d_model  # K 和 V 前缀
    
    print(f"全参数微调: {full_params:,} 参数")
    print(f"Prefix Tuning: {prefix_params:,} 参数")
    print(f"参数比例: {prefix_params/full_params*100:.4f}%")

prefix_params_comparison()
```

### 2.2 Adapter

```python
import numpy as np

class AdapterLayer:
    """Adapter 层"""
    
    def __init__(self, d_model, bottleneck=64, activation="relu"):
        """
        参数:
            d_model: 模型维度
            bottleneck: 瓶颈维度
            activation: 激活函数
        """
        self.d_model = d_model
        self.bottleneck = bottleneck
        
        # 下投影
        self.down_proj = np.random.randn(d_model, bottleneck) * 0.01
        # 上投影
        self.up_proj = np.random.randn(bottleneck, d_model) * 0.01
        
        self.activation = activation
    
    def forward(self, x):
        """
        前向传播
        
        output = x + Up(Activation(Down(x)))
        """
        # 下投影
        down = np.matmul(x, self.down_proj)
        
        # 激活
        if self.activation == "relu":
            activated = np.maximum(0, down)
        elif self.activation == "gelu":
            activated = 0.5 * down * (1 + np.tanh(np.sqrt(2/np.pi) * (down + 0.044715 * down**3)))
        
        # 上投影
        up = np.matmul(activated, self.up_proj)
        
        # 残差连接
        return x + up


class AdapterTransformerBlock:
    """带 Adapter 的 Transformer 块"""
    
    def __init__(self, d_model, n_heads, d_ff, adapter_bottleneck=64):
        # Self-Attention
        self.attention = SelfAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        
        # Adapter after attention
        self.adapter1 = AdapterLayer(d_model, adapter_bottleneck)
        
        # FFN
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
        
        # Adapter after FFN
        self.adapter2 = AdapterLayer(d_model, adapter_bottleneck)
    
    def forward(self, x):
        # Self-Attention + Adapter
        residual = x
        x = self.norm1.forward(x)
        x = self.attention.forward(x)
        x = self.adapter1.forward(x)  # Adapter 在这里
        x = residual + x
        
        # FFN + Adapter
        residual = x
        x = self.norm2.forward(x)
        x = self.ffn.forward(x)
        x = self.adapter2.forward(x)  # Adapter 在这里
        x = residual + x
        
        return x


# 简化的辅助类
class LayerNorm:
    def __init__(self, d_model):
        self.weight = np.ones(d_model)
        self.bias = np.zeros(d_model)
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.weight * (x - mean) / np.sqrt(var + 1e-6) + self.bias

class SelfAttention:
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = np.random.randn(d_model, d_model) * 0.02
        self.k_proj = np.random.randn(d_model, d_model) * 0.02
        self.v_proj = np.random.randn(d_model, d_model) * 0.02
        self.o_proj = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, x):
        batch, seq, _ = x.shape
        Q = np.matmul(x, self.q_proj).reshape(batch, seq, self.n_heads, self.d_k).transpose(0,2,1,3)
        K = np.matmul(x, self.k_proj).reshape(batch, seq, self.n_heads, self.d_k).transpose(0,2,1,3)
        V = np.matmul(x, self.v_proj).reshape(batch, seq, self.n_heads, self.d_k).transpose(0,2,1,3)
        scores = np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(self.d_k)
        attn = np.exp(scores - np.max(scores, -1, keepdims=True))
        attn = attn / np.sum(attn, -1, keepdims=True)
        out = np.matmul(attn, V).transpose(0,2,1,3).reshape(batch, seq, -1)
        return np.matmul(out, self.o_proj)

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.up = np.random.randn(d_model, d_ff) * 0.02
        self.down = np.random.randn(d_ff, d_model) * 0.02
    
    def forward(self, x):
        return np.matmul(np.maximum(0, np.matmul(x, self.up)), self.down)
```

### 2.3 QLoRA

```python
import numpy as np

class QLoRALayer:
    """QLoRA: LoRA + 4-bit 量化"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16, 
                 bits=4, compute_dtype=np.float16):
        """
        参数:
            in_features: 输入维度
            out_features: 输出维度
            rank: LoRA 秩
            alpha: 缩放因子
            bits: 量化位数（4 或 8）
            compute_dtype: 计算精度
        """
        self.bits = bits
        self.compute_dtype = compute_dtype
        
        # 4-bit 量化权重
        self.quantized_weight = None  # 量化后的权重
        self.weight_absmax = None      # 量化参数
        
        # LoRA（保持高精度）
        self.lora_A = np.random.randn(in_features, rank).astype(compute_dtype) * 0.01
        self.lora_B = np.zeros((rank, out_features), dtype=compute_dtype)
        
        self.scaling = alpha / rank
    
    def quantize(self, weight):
        """4-bit 量化（简化版）"""
        # NF4 量化的简化实现
        absmax = np.max(np.abs(weight))
        normalized = weight / absmax
        
        # 简单的均匀量化
        quantized = np.clip(np.round(normalized * (2**(self.bits-1) - 1)), 
                           -(2**(self.bits-1)), 
                           2**(self.bits-1) - 1)
        
        return quantized.astype(np.int8), absmax
    
    def dequantize(self, quantized, absmax):
        """反量化"""
        return quantized.astype(self.compute_dtype) * absmax / (2**(self.bits-1) - 1)
    
    def forward(self, x):
        """前向传播"""
        # 反量化权重
        W = self.dequantize(self.quantized_weight, self.weight_absmax)
        
        # 原始输出
        original = np.matmul(x.astype(self.compute_dtype), W)
        
        # LoRA 输出
        lora = np.matmul(x.astype(self.compute_dtype), 
                        np.matmul(self.lora_A, self.lora_B))
        
        return original + self.scaling * lora


# 显存对比
def memory_comparison():
    d = 4096
    rank = 8
    
    print("显存占用对比（单层线性层）:")
    
    # FP16 全参数
    fp16_params = d * d
    fp16_memory = fp16_params * 2  # 2 bytes per param
    print(f"FP16 全参数: {fp16_memory / 1024**2:.1f} MB")
    
    # FP16 LoRA
    lora_params = 2 * d * rank
    lora_memory = lora_params * 2
    print(f"FP16 LoRA:   {lora_memory / 1024**2:.2f} MB")
    
    # 4-bit + FP16 LoRA
    quant_memory = d * d * 0.5 + lora_params * 2  # 0.5 bytes for 4-bit
    print(f"4-bit QLoRA: {quant_memory / 1024**2:.1f} MB")
    
    print(f"\nQLoRA 节省: {(1 - quant_memory/fp16_memory)*100:.1f}% 显存")

memory_comparison()
```

---

## 3. 使用 Hugging Face PEFT

### 3.1 快速开始

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 加载基础模型
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                        # LoRA 秩
    lora_alpha=16,              # 缩放因子
    lora_dropout=0.1,           # Dropout
    target_modules=["q_proj", "v_proj"],  # 应用 LoRA 的模块
    bias="none",
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 打印可训练参数
model.print_trainable_parameters()
# 输出: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%

# 训练
# from transformers import Trainer
# trainer = Trainer(model=model, ...)
# trainer.train()

# 保存 LoRA 权重
model.save_pretrained("./lora_weights")

# 合并权重
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
```

### 3.2 不同 PEFT 方法配置

```python
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    AdaLoraConfig,
    IA3Config,
)

# ========== LoRA ==========
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
)

# ========== AdaLoRA（自适应秩）==========
adalora_config = AdaLoraConfig(
    init_r=12,              # 初始秩
    target_r=4,             # 目标秩
    beta1=0.85,
    beta2=0.85,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    target_modules=["q_proj", "v_proj"],
)

# ========== Prefix Tuning ==========
prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,  # 前缀长度
    prefix_projection=True,
)

# ========== Prompt Tuning ==========
prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="你是一个有帮助的AI助手。",
    tokenizer_name_or_path=model_name,
)

# ========== IA³ ==========
ia3_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
)
```

---

## 💻 完整微调示例

```python
"""
完整的 LoRA 微调示例
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def finetune_with_lora():
    """使用 LoRA 微调模型"""
    
    # 加载模型
    model_name = "Qwen/Qwen2-1.5B"  # 小模型示例
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载数据
    dataset = load_dataset("imdb", split="train[:1000]")
    
    # 预处理
    def preprocess(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )
    
    tokenized = dataset.map(preprocess, batched=True)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        logging_steps=100,
        save_steps=500,
        fp16=True,
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # 训练
    trainer.train()
    
    # 保存
    model.save_pretrained("./my_lora_model")
    
    return model

# 运行
if __name__ == "__main__":
    model = finetune_with_lora()
    print("训练完成!")
```

---

## 📝 本章小结

### 核心要点

1. **LoRA**：低秩分解，最流行的 PEFT 方法
2. **Prefix Tuning**：添加可学习前缀向量
3. **Adapter**：在层间插入小型适配器
4. **QLoRA**：结合量化的 LoRA，大幅节省显存

### 方法选择指南

| 方法 | 参数量 | 显存 | 训练速度 | 推荐场景 |
|------|--------|------|----------|----------|
| LoRA | 0.1% | 低 | 快 | 通用首选 |
| QLoRA | 0.1% | 最低 | 中 | 显存受限 |
| Prefix Tuning | 0.1% | 低 | 快 | 生成任务 |
| Adapter | 1% | 中 | 中 | 模块化部署 |

---

<div align="center">

[⬅️ 上一章](../chapter16-mainstream-llms/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter18-instruction-tuning-rlhf/README.md)

</div>
