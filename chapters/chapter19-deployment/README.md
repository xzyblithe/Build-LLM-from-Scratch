# 第19章：模型部署与推理优化

<div align="center">

[⬅️ 上一章](../chapter18-instruction-tuning-&-rlhf/README.md) | [返回目录](../README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 掌握模型量化技术（INT8/INT4）
- ✅ 理解 KV Cache 原理与实现
- ✅ 了解 Flash Attention 优化
- ✅ 使用 vLLM、TensorRT-LLM 等推理框架
- ✅ 部署高效的模型推理服务

---

## 🎯 推理优化概览

### 为什么需要推理优化？

```
大模型推理挑战：

1. 显存占用大
   - LLaMA-7B FP16: ~14GB 显存
   - LLaMA-70B FP16: ~140GB 显存

2. 推理延迟高
   - 自回归生成，逐 token 输出
   - 每个位置都要计算完整的注意力

3. 吞吐量低
   - 批处理效率低
   - 内存带宽瓶颈

优化方向：
├── 量化：减少显存占用
├── KV Cache：避免重复计算
├── Flash Attention：优化注意力计算
├── 连续批处理：提高吞吐量
└── 模型并行：分布式推理
```

---

## 1. 模型量化

### 1.1 量化原理

```
量化（Quantization）：降低模型精度

FP16 → INT8 → INT4

量化方式：
1. 训练后量化（PTQ）
   - 直接对训练好的模型量化
   - 简单快速，可能有精度损失

2. 量化感知训练（QAT）
   - 训练时模拟量化
   - 精度损失小，但训练成本高

量化公式：
  Q(x) = round(x / scale) + zero_point
  
反量化：
  D(q) = (q - zero_point) * scale
```

### 1.2 INT8 量化实现

```python
import numpy as np

class INT8Quantizer:
    """INT8 对称量化器"""
    
    def __init__(self):
        self.scale = None
    
    def quantize(self, weight):
        """
        量化权重到 INT8
        
        对称量化：zero_point = 0
        Q(x) = round(x / scale)
        scale = max(|x|) / 127
        """
        # 计算 scale
        abs_max = np.max(np.abs(weight))
        self.scale = abs_max / 127.0
        
        # 量化
        quantized = np.clip(np.round(weight / self.scale), -128, 127)
        
        return quantized.astype(np.int8)
    
    def dequantize(self, quantized):
        """反量化回 FP32"""
        return quantized.astype(np.float32) * self.scale
    
    def quantize_dynamic(self, activation):
        """
        动态量化（激活值）
        每次推理时重新计算 scale
        """
        abs_max = np.max(np.abs(activation))
        scale = abs_max / 127.0
        quantized = np.clip(np.round(activation / scale), -128, 127)
        return quantized.astype(np.int8), scale


class QuantizedLinear:
    """量化线性层"""
    
    def __init__(self, weight, bias=None):
        """
        参数:
            weight: FP32 权重
            bias: 偏置
        """
        # 量化权重
        self.quantizer = INT8Quantizer()
        self.quantized_weight = self.quantizer.quantize(weight)
        self.scale = self.quantizer.scale
        
        # 偏置保持高精度
        self.bias = bias
    
    def forward(self, x):
        """
        前向传播
        
        1. 动态量化输入
        2. INT8 矩阵乘法
        3. 反量化输出
        """
        # 动态量化输入
        x_quant, x_scale = INT8Quantizer().quantize_dynamic(x)
        
        # 反量化后计算（实际部署中会用 INT8 计算）
        w_dequant = self.quantizer.dequantize(self.quantized_weight)
        x_dequant = x_quant.astype(np.float32) * x_scale
        
        output = np.matmul(x_dequant, w_dequant)
        
        if self.bias is not None:
            output += self.bias
        
        return output


# 量化效果对比
def quantization_comparison():
    """量化效果对比"""
    
    # 模拟权重
    d = 4096
    weight = np.random.randn(d, d).astype(np.float32)
    
    print("量化效果对比:")
    print("-" * 50)
    
    # FP32
    fp32_size = weight.nbytes
    print(f"FP32: {fp32_size / 1024**2:.2f} MB")
    
    # FP16
    fp16_size = weight.astype(np.float16).nbytes
    print(f"FP16: {fp16_size / 1024**2:.2f} MB (压缩 {fp32_size/fp16_size:.1f}x)")
    
    # INT8
    quantizer = INT8Quantizer()
    int8_weight = quantizer.quantize(weight)
    int8_size = int8_weight.nbytes
    print(f"INT8: {int8_size / 1024**2:.2f} MB (压缩 {fp32_size/int8_size:.1f}x)")
    
    # INT4 (模拟)
    int4_size = int8_size // 2
    print(f"INT4: ~{int4_size / 1024**2:.2f} MB (压缩 {fp32_size/int4_size:.1f}x)")

quantization_comparison()
```

### 1.3 GPTQ 与 AWQ

```python
"""
高级量化方法

GPTQ (Gradient-based Post-Training Quantization):
- 基于二阶信息的训练后量化
- 逐层量化，使用 Hessian 矩阵
- 适合 INT4 量化

AWQ (Activation-aware Weight Quantization):
- 保护重要权重通道
- 基于激活值分析权重重要性
- 精度损失更小
"""

import numpy as np

class GPTQQuantizer:
    """简化版 GPTQ 量化器"""
    
    def __init__(self, bits=4, group_size=128):
        """
        参数:
            bits: 量化位数
            group_size: 分组大小
        """
        self.bits = bits
        self.group_size = group_size
    
    def quantize(self, weight, hessian=None):
        """
        GPTQ 量化
        
        使用 Hessian 信息找到最优量化
        """
        # 简化实现：实际 GPTQ 更复杂
        n_groups = weight.shape[0] // self.group_size
        quantized = np.zeros_like(weight, dtype=np.int8)
        scales = []
        
        for i in range(n_groups):
            start = i * self.group_size
            end = (i + 1) * self.group_size
            group_weight = weight[start:end]
            
            # 计算该组的 scale
            scale = np.max(np.abs(group_weight)) / (2 ** (self.bits - 1) - 1)
            scales.append(scale)
            
            # 量化
            quantized[start:end] = np.clip(
                np.round(group_weight / scale),
                -(2 ** (self.bits - 1)),
                2 ** (self.bits - 1) - 1
            )
        
        return quantized, np.array(scales)


class AWQQuantizer:
    """简化版 AWQ 量化器"""
    
    def __init__(self, bits=4, group_size=128):
        self.bits = bits
        self.group_size = group_size
    
    def find_important_channels(self, weight, activations):
        """
        找出重要通道
        
        基于激活值幅度分析权重重要性
        """
        # 计算激活值的重要性
        activation_importance = np.mean(np.abs(activations), axis=0)
        
        # 归一化
        importance = activation_importance / np.max(activation_importance)
        
        return importance
    
    def quantize_with_importance(self, weight, importance):
        """
        根据重要性调整量化策略
        """
        # 对重要通道使用更细的量化
        # 简化实现
        scale = np.max(np.abs(weight)) / (2 ** (self.bits - 1) - 1)
        quantized = np.clip(np.round(weight / scale), 
                           -(2 ** (self.bits - 1)),
                           2 ** (self.bits - 1) - 1)
        return quantized.astype(np.int8), scale


# 使用 bitsandbytes 量化
def quantize_with_bitsandbytes():
    """使用 bitsandbytes 进行 4-bit 量化"""
    code = """
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

print(f"模型加载成功，显存占用大幅减少")
"""
    print("bitsandbytes 4-bit 量化示例:")
    print(code)


print("\n高级量化方法:")
print("-" * 50)
quantize_with_bitsandbytes()
```

---

## 2. KV Cache

### 2.1 KV Cache 原理

```
自回归生成的计算问题：

生成 "我喜欢学习人工智能" 的过程：
输入: [我喜欢]
输出: [学]  ← 需要计算 [我]、[喜]、[欢] 的注意力

输入: [我喜欢学]
输出: [习]  ← 又要计算 [我]、[喜]、[欢]、[学] 的注意力

问题：每个新 token 都要重新计算之前所有 token 的 K、V

KV Cache 解决方案：
- 第一次计算时，缓存 K 和 V
- 后续推理直接使用缓存的 K、V
- 只计算新 token 的 K、V

显存节省：
无缓存: O(n²) 次计算
有缓存: O(n) 次计算
```

### 2.2 KV Cache 实现

```python
import numpy as np

class KVCache:
    """KV Cache 管理"""
    
    def __init__(self, n_layers, n_heads, d_k, max_seq_len=2048, dtype=np.float16):
        """
        参数:
            n_layers: 层数
            n_heads: 头数
            d_k: 头维度
            max_seq_len: 最大序列长度
            dtype: 数据类型
        """
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # 预分配缓存 (batch=1)
        self.k_cache = [
            np.zeros((1, n_heads, max_seq_len, d_k), dtype=dtype)
            for _ in range(n_layers)
        ]
        self.v_cache = [
            np.zeros((1, n_heads, max_seq_len, d_k), dtype=dtype)
            for _ in range(n_layers)
        ]
        
        # 当前序列长度
        self.seq_len = 0
    
    def update(self, layer_idx, k_new, v_new):
        """
        更新缓存
        
        参数:
            layer_idx: 层索引
            k_new: 新的 K (batch, n_heads, 1, d_k)
            v_new: 新的 V (batch, n_heads, 1, d_k)
        """
        # 追加到缓存
        self.k_cache[layer_idx][:, :, self.seq_len:self.seq_len+1, :] = k_new
        self.v_cache[layer_idx][:, :, self.seq_len:self.seq_len+1, :] = v_new
    
    def get(self, layer_idx):
        """
        获取缓存
        
        返回: (K, V) 从位置 0 到当前位置
        """
        return (
            self.k_cache[layer_idx][:, :, :self.seq_len+1, :],
            self.v_cache[layer_idx][:, :, :self.seq_len+1, :]
        )
    
    def advance(self, n=1):
        """前进 n 个位置"""
        self.seq_len += n
    
    def reset(self):
        """重置缓存"""
        self.seq_len = 0
    
    def get_memory_size(self):
        """获取缓存占用内存"""
        # 每层有两个缓存，每个大小为 n_heads * max_seq_len * d_k * 2 bytes
        bytes_per_cache = self.n_heads * self.max_seq_len * self.d_k * 2
        total_bytes = 2 * self.n_layers * bytes_per_cache
        return total_bytes / (1024 ** 2)  # MB


class CachedAttention:
    """带 KV Cache 的注意力层"""
    
    def __init__(self, d_model, n_heads, max_seq_len=2048):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 投影矩阵
        self.q_proj = np.random.randn(d_model, d_model).astype(np.float16) * 0.02
        self.k_proj = np.random.randn(d_model, d_model).astype(np.float16) * 0.02
        self.v_proj = np.random.randn(d_model, d_model).astype(np.float16) * 0.02
        self.o_proj = np.random.randn(d_model, d_model).astype(np.float16) * 0.02
    
    def forward(self, x, kv_cache=None, layer_idx=0, use_cache=True):
        """
        前向传播
        
        参数:
            x: 输入 (batch, seq_len, d_model)
            kv_cache: KV Cache 对象
            layer_idx: 层索引
            use_cache: 是否使用缓存
        """
        batch_size, seq_len, _ = x.shape
        
        # 只处理最后一个 token（推理时）
        if use_cache and kv_cache is not None and seq_len == 1:
            # Prefill 阶段已缓存，现在只需处理新 token
            return self._forward_cached(x, kv_cache, layer_idx)
        else:
            # Prefill 阶段：处理整个序列
            return self._forward_prefill(x, kv_cache, layer_idx, use_cache)
    
    def _forward_prefill(self, x, kv_cache, layer_idx, use_cache):
        """Prefill: 处理完整序列"""
        batch_size, seq_len, _ = x.shape
        
        # QKV 投影
        Q = np.matmul(x, self.q_proj)
        K = np.matmul(x, self.k_proj)
        V = np.matmul(x, self.v_proj)
        
        # 分头
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 更新缓存
        if use_cache and kv_cache is not None:
            for i in range(seq_len):
                kv_cache.update(layer_idx, K[:, :, i:i+1, :], V[:, :, i:i+1, :])
                kv_cache.advance()
        
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
    
    def _forward_cached(self, x, kv_cache, layer_idx):
        """Decode: 使用缓存，只处理新 token"""
        batch_size = x.shape[0]
        
        # QKV 投影
        Q = np.matmul(x, self.q_proj)
        K = np.matmul(x, self.k_proj)
        V = np.matmul(x, self.v_proj)
        
        # 分头
        Q = Q.reshape(batch_size, 1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, 1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, 1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 更新缓存
        kv_cache.update(layer_idx, K, V)
        
        # 获取完整 K、V
        K_cached, V_cached = kv_cache.get(layer_idx)
        
        # 注意力（只计算新 token 的 Q 与所有 K、V）
        scores = np.matmul(Q, K_cached.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # Softmax
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        # 输出
        output = np.matmul(attn_weights, V_cached)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, 1, -1)
        
        # 推进位置
        kv_cache.advance()
        
        return np.matmul(output, self.o_proj)


# KV Cache 显存计算
def kv_cache_memory():
    """计算 KV Cache 显存占用"""
    
    print("\nKV Cache 显存占用:")
    print("-" * 60)
    
    models = [
        ("LLaMA-7B", 32, 32, 128, 4096),
        ("LLaMA-13B", 40, 40, 128, 4096),
        ("LLaMA-70B", 80, 64, 128, 4096),
    ]
    
    for name, n_layers, n_heads, d_k, max_seq in models:
        kv = KVCache(n_layers, n_heads, d_k, max_seq)
        print(f"{name}: {kv.get_memory_size():.1f} MB (seq_len={max_seq})")

kv_cache_memory()
```

---

## 3. 推理框架

### 3.1 vLLM

```python
"""
vLLM: 高吞吐量推理框架

核心特性：
1. PagedAttention
   - 将 KV Cache 分页管理
   - 类似操作系统的虚拟内存
   - 支持连续批处理

2. 连续批处理
   - 动态调度请求
   - 提高吞吐量

3. 优化的 CUDA 内核
"""

# vLLM 使用示例
vllm_example = """
# 安装
pip install vllm

# 基本使用
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256
)

# 批量推理
prompts = [
    "什么是机器学习？",
    "解释一下深度学习的原理。",
    "自然语言处理有哪些应用？"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Response: {output.outputs[0].text}")
    print()

# API 服务
# python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-7b-hf
"""

print("vLLM 使用示例:")
print(vllm_example)
```

### 3.2 TensorRT-LLM

```python
"""
TensorRT-LLM: NVIDIA 的高性能推理框架

核心特性：
1. 针对 NVIDIA GPU 优化
2. 支持多种量化方法
3. 内核融合优化
4. 多 GPU 推理
"""

tensorrt_example = """
# 安装
pip install tensorrt-llm

# 转换模型
python convert_checkpoint.py --model_dir ./llama-7b \\
    --output_dir ./trt_llama_7b \\
    --tp 1

# 构建引擎
trtllm-build --checkpoint_dir ./trt_llama_7b \\
    --output_dir ./trt_engines \\
    --max_batch_size 8 \\
    --max_input_len 1024 \\
    --max_output_len 256

# 运行推理
from tensorrt_llm import LLM, SamplingParams

llm = LLM(engine_dir="./trt_engines")
outputs = llm.generate(["Hello, how are you?"], max_tokens=50)
"""

print("\nTensorRT-LLM 示例:")
print(tensorrt_example)
```

### 3.3 简单的推理服务

```python
"""
简单的模型推理服务
"""
import numpy as np
from typing import List, Optional
import json

class InferenceServer:
    """推理服务器"""
    
    def __init__(self, model_path, max_batch_size=8, max_seq_len=2048):
        """
        参数:
            model_path: 模型路径
            max_batch_size: 最大批次大小
            max_seq_len: 最大序列长度
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        
        # 加载模型（简化）
        self.model = SimpleLLM()
        self.tokenizer = SimpleTokenizer()
        
        # 请求队列
        self.request_queue = []
        
        # 统计
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'avg_latency': 0
        }
    
    def generate(self, prompt: str, max_tokens: int = 100, 
                 temperature: float = 0.7, top_p: float = 0.95) -> str:
        """
        生成回复
        
        参数:
            prompt: 输入提示
            max_tokens: 最大生成长度
            temperature: 温度参数
            top_p: Top-p 采样参数
        """
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)
        
        # 生成
        output_ids = self.model.generate(
            input_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Detokenize
        output_text = self.tokenizer.decode(output_ids)
        
        return output_text
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成"""
        # 简化：逐个处理
        # 实际中应该使用连续批处理
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results
    
    def stream_generate(self, prompt: str, **kwargs):
        """流式生成"""
        input_ids = self.tokenizer.encode(prompt)
        
        for i in range(kwargs.get('max_tokens', 100)):
            # 生成下一个 token
            next_token = self.model.generate_next_token(input_ids, **kwargs)
            input_ids.append(next_token)
            
            # 返回 token
            yield self.tokenizer.decode([next_token])


# 简化的模型和分词器
class SimpleLLM:
    """简化的语言模型"""
    
    def __init__(self, vocab_size=10000, d_model=512):
        self.vocab_size = vocab_size
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
    
    def generate(self, input_ids, max_tokens=100, temperature=0.7, top_p=0.95):
        """生成序列"""
        output = list(input_ids)
        
        for _ in range(max_tokens):
            next_token = self.generate_next_token(output, temperature, top_p)
            output.append(next_token)
        
        return output
    
    def generate_next_token(self, input_ids, temperature=0.7, top_p=0.95):
        """生成下一个 token"""
        # 简化：随机采样
        logits = np.random.randn(self.vocab_size)
        
        # Temperature
        logits = logits / temperature
        
        # Top-p 采样
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))
        
        # 找到 top-p 边界
        top_p_indices = sorted_indices[cumulative_probs <= top_p]
        
        if len(top_p_indices) == 0:
            top_p_indices = sorted_indices[:1]
        
        # 采样
        return np.random.choice(top_p_indices)


class SimpleTokenizer:
    """简化的分词器"""
    
    def __init__(self):
        self.vocab = {chr(i): i for i in range(100)}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(c, 0) for c in text]
    
    def decode(self, ids: List[int]) -> str:
        return ''.join(self.inv_vocab.get(i, '?') for i in ids)


# FastAPI 集成示例
fastapi_example = """
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
server = InferenceServer("./model")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerateRequest):
    result = server.generate(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    return {"response": result}

@app.post("/stream")
async def stream(request: GenerateRequest):
    async def generate_stream():
        for token in server.stream_generate(request.prompt):
            yield f"data: {token}\\n\\n"
    return StreamingResponse(generate_stream())

# 运行: uvicorn server:app --host 0.0.0.0 --port 8000
"""

print("\nFastAPI 服务示例:")
print(fastapi_example)
```

---

## 4. 性能优化技巧

### 4.1 推理优化清单

```python
"""
大模型推理优化技巧
"""

optimization_tips = """
推理优化技巧清单：

1. 模型优化
   ├── 量化: INT8/INT4 减少显存和计算
   ├── 剪枝: 移除冗余参数
   └── 蒸馏: 小模型学习大模型

2. 计算优化
   ├── Flash Attention: 减少内存访问
   ├── 算子融合: 减少内核启动开销
   └── KV Cache: 避免重复计算

3. 调度优化
   ├── 连续批处理: 动态合并请求
   ├── 投机解码: 加速生成
   └── 前缀缓存: 缓存共享前缀

4. 硬件优化
   ├── GPU 并行: 多卡推理
   ├── CPU 卸载: 层间卸载
   └── NVLink: 节点间通信

5. 服务优化
   ├── 请求队列: 批处理等待
   ├── 模型预热: 首次请求加速
   └── 负载均衡: 多实例部署
"""

print(optimization_tips)


# 性能指标
def benchmark_metrics():
    """推理性能指标"""
    
    print("\n推理性能指标:")
    print("-" * 60)
    
    print(f"{'指标':<20} {'说明':<30} {'目标值'}")
    print("-" * 60)
    
    metrics = [
        ("首 Token 延迟 (TTFT)", "第一个 token 生成时间", "< 500ms"),
        ("Token 生成延迟", "每个 token 生成时间", "< 50ms"),
        ("吞吐量", "每秒生成 token 数", "> 100 tokens/s"),
        ("显存占用", "模型和 KV Cache 占用", "根据模型大小"),
        ("并发数", "同时处理的请求数", "> 16"),
    ]
    
    for name, desc, target in metrics:
        print(f"{name:<20} {desc:<30} {target}")

benchmark_metrics()
```

---

## 5. 部署架构

### 5.1 生产部署架构

```
生产环境部署架构:

┌─────────────────────────────────────────────────────────────┐
│                        负载均衡器                            │
│                   (Nginx / Kong / AWS ALB)                  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   API Gateway │     │   API Gateway │     │   API Gateway │
│   (FastAPI)   │     │   (FastAPI)   │     │   (FastAPI)   │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│   vLLM /      │     │   vLLM /      │     │   vLLM /      │
│   TRT-LLM     │     │   TRT-LLM     │     │   TRT-LLM     │
│   GPU 0       │     │   GPU 1       │     │   GPU 2       │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │    Redis      │
                    │  (请求队列)    │
                    └───────────────┘
```

### 5.2 部署配置示例

```python
"""
生产环境部署配置
"""

deployment_config = """
# docker-compose.yml
version: '3.8'

services:
  api:
    image: my-llm-api:latest
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/llama-7b
      - MAX_BATCH_SIZE=32
      - MAX_SEQ_LEN=4096
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api

# Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-inference
  template:
    metadata:
      labels:
        app: llm-inference
    spec:
      containers:
      - name: api
        image: my-llm-api:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        ports:
        - containerPort: 8000
"""

print("部署配置示例:")
print(deployment_config)
```

---

## 💻 完整推理服务示例

```python
"""
完整的模型推理服务
"""
import numpy as np
from typing import List, Dict, Optional, AsyncGenerator
from dataclasses import dataclass
import asyncio
import time

@dataclass
class GenerationConfig:
    """生成配置"""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1


class LLMInferenceEngine:
    """LLM 推理引擎"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        参数:
            model_path: 模型路径
            device: 设备 (cuda/cpu)
        """
        self.device = device
        
        # 加载模型（简化）
        self.vocab_size = 10000
        self.d_model = 512
        self.n_layers = 8
        self.n_heads = 8
        
        # 初始化 KV Cache
        self.kv_cache = KVCache(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_k=self.d_model // self.n_heads,
            max_seq_len=4096
        )
        
        # 性能统计
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_time': 0
        }
    
    def generate(
        self,
        input_ids: List[int],
        config: GenerationConfig = None
    ) -> List[int]:
        """
        生成回复
        
        参数:
            input_ids: 输入 token IDs
            config: 生成配置
        
        返回:
            输出 token IDs
        """
        if config is None:
            config = GenerationConfig()
        
        self.stats['total_requests'] += 1
        start_time = time.time()
        
        # 初始化
        output_ids = list(input_ids)
        self.kv_cache.reset()
        
        # 生成
        for _ in range(config.max_tokens):
            next_token = self._generate_next_token(
                output_ids, config
            )
            
            # EOS 检查
            if next_token == 2:  # EOS token
                break
            
            output_ids.append(next_token)
            self.stats['total_tokens'] += 1
        
        # 统计
        self.stats['total_time'] += time.time() - start_time
        
        return output_ids[len(input_ids):]  # 只返回生成部分
    
    async def generate_stream(
        self,
        input_ids: List[int],
        config: GenerationConfig = None
    ) -> AsyncGenerator[int, None]:
        """流式生成"""
        if config is None:
            config = GenerationConfig()
        
        output_ids = list(input_ids)
        self.kv_cache.reset()
        
        for _ in range(config.max_tokens):
            next_token = self._generate_next_token(output_ids, config)
            
            if next_token == 2:
                break
            
            output_ids.append(next_token)
            yield next_token
            
            # 异步让步
            await asyncio.sleep(0)
    
    def _generate_next_token(
        self,
        input_ids: List[int],
        config: GenerationConfig
    ) -> int:
        """生成下一个 token"""
        # 简化：随机采样
        logits = np.random.randn(self.vocab_size)
        
        # Temperature
        logits = logits / config.temperature
        
        # Repetition penalty
        if len(input_ids) > 0:
            for prev_token in set(input_ids):
                logits[prev_token] /= config.repetition_penalty
        
        # Top-k 过滤
        top_k_indices = np.argsort(logits)[-config.top_k:]
        mask = np.ones(self.vocab_size, dtype=bool)
        mask[top_k_indices] = False
        logits[mask] = -float('inf')
        
        # Top-p 采样
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        probs = np.exp(sorted_logits - np.max(sorted_logits))
        probs = probs / probs.sum()
        
        cumulative_probs = np.cumsum(probs)
        top_p_mask = cumulative_probs <= config.top_p
        
        if not any(top_p_mask):
            top_p_mask[0] = True
        
        top_p_indices = sorted_indices[top_p_mask]
        top_p_probs = probs[top_p_mask]
        top_p_probs = top_p_probs / top_p_probs.sum()
        
        # 采样
        return np.random.choice(top_p_indices, p=top_p_probs)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if self.stats['total_requests'] == 0:
            return self.stats
        
        avg_latency = self.stats['total_time'] / self.stats['total_requests']
        throughput = self.stats['total_tokens'] / self.stats['total_time']
        
        return {
            **self.stats,
            'avg_latency_ms': avg_latency * 1000,
            'throughput_tokens_per_sec': throughput
        }


# 使用示例
async def main():
    """使用示例"""
    
    # 初始化引擎
    engine = LLMInferenceEngine("./model")
    
    # 配置
    config = GenerationConfig(
        max_tokens=50,
        temperature=0.7,
        top_p=0.95
    )
    
    # 同步生成
    input_ids = [1, 2, 3, 4, 5]  # 简化输入
    output_ids = engine.generate(input_ids, config)
    print(f"生成 token 数: {len(output_ids)}")
    
    # 流式生成
    print("\n流式生成:")
    async for token in engine.generate_stream(input_ids, config):
        print(f"Token: {token}", end=" ", flush=True)
    
    # 统计
    print(f"\n\n统计信息: {engine.get_stats()}")

# 运行
# asyncio.run(main())

print("LLM 推理引擎示例完成")
```

---

## 📝 本章小结

### 核心要点

1. **量化**：INT8/INT4 减少模型大小和显存占用
2. **KV Cache**：缓存注意力计算的 K、V，避免重复计算
3. **推理框架**：vLLM、TensorRT-LLM 提供高效推理
4. **服务部署**：FastAPI + 容器化 + 负载均衡

### 性能优化清单

| 优化方向 | 技术 | 效果 |
|----------|------|------|
| 显存 | 量化、KV Cache | 50%~75% 减少 |
| 速度 | Flash Attention、算子融合 | 2~4x 加速 |
| 吞吐 | 连续批处理、vLLM | 5~10x 提升 |
| 延迟 | 投机解码、前缀缓存 | 30%~50% 降低 |

---

<div align="center">

[⬅️ 上一章](../chapter18-instruction-tuning-&-rlhf/README.md) | [返回目录](../README.md)

**🎉 恭喜完成全部教程！**

</div>
