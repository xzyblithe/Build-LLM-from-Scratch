"""
第19章：模型部署与推理优化
量化、KV Cache、推理服务等
"""
import numpy as np
from typing import List, Optional, Dict, AsyncGenerator
from dataclasses import dataclass
import time


# ========== 1. 量化 ==========

class INT8Quantizer:
    """INT8 量化器"""
    
    def __init__(self):
        self.scale = None
    
    def quantize(self, weight: np.ndarray) -> tuple:
        """
        量化到 INT8
        
        返回: (量化权重, scale)
        """
        abs_max = np.max(np.abs(weight))
        self.scale = abs_max / 127.0
        
        quantized = np.clip(np.round(weight / self.scale), -128, 127)
        return quantized.astype(np.int8), self.scale
    
    def dequantize(self, quantized: np.ndarray, scale: float) -> np.ndarray:
        """反量化"""
        return quantized.astype(np.float32) * scale


class QuantizedLinear:
    """量化线性层"""
    
    def __init__(self, weight: np.ndarray, bias: np.ndarray = None):
        self.quantizer = INT8Quantizer()
        self.quantized_weight, self.scale = self.quantizer.quantize(weight)
        self.bias = bias
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        # 反量化
        weight = self.quantizer.dequantize(self.quantized_weight, self.scale)
        
        # 计算
        output = np.matmul(x, weight)
        if self.bias is not None:
            output += self.bias
        
        return output


# ========== 2. KV Cache ==========

class KVCache:
    """KV Cache 管理"""
    
    def __init__(self, n_layers: int, n_heads: int, d_k: int, 
                 max_seq_len: int = 2048):
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # 预分配缓存
        self.k_cache = [
            np.zeros((1, n_heads, max_seq_len, d_k), dtype=np.float16)
            for _ in range(n_layers)
        ]
        self.v_cache = [
            np.zeros((1, n_heads, max_seq_len, d_k), dtype=np.float16)
            for _ in range(n_layers)
        ]
        
        self.seq_len = 0
    
    def update(self, layer_idx: int, k: np.ndarray, v: np.ndarray):
        """更新缓存"""
        self.k_cache[layer_idx][:, :, self.seq_len:self.seq_len+1, :] = k
        self.v_cache[layer_idx][:, :, self.seq_len:self.seq_len+1, :] = v
    
    def get(self, layer_idx: int) -> tuple:
        """获取缓存"""
        return (
            self.k_cache[layer_idx][:, :, :self.seq_len+1, :],
            self.v_cache[layer_idx][:, :, :self.seq_len+1, :]
        )
    
    def advance(self, n: int = 1):
        """前进 n 个位置"""
        self.seq_len += n
    
    def reset(self):
        """重置"""
        self.seq_len = 0
    
    def memory_size_mb(self) -> float:
        """计算内存占用"""
        bytes_per_cache = self.k_cache[0].nbytes
        total = 2 * self.n_layers * bytes_per_cache
        return total / (1024 ** 2)


# ========== 3. 推理服务 ==========

@dataclass
class GenerationConfig:
    """生成配置"""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50


class SimpleTokenizer:
    """简单分词器"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
    
    def encode(self, text: str) -> List[int]:
        """编码（简化）"""
        return [hash(c) % self.vocab_size for c in text[:100]]
    
    def decode(self, ids: List[int]) -> str:
        """解码（简化）"""
        return ''.join(chr(i % 128 + 32) for i in ids)


class LLMInferenceEngine:
    """LLM 推理引擎"""
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 512,
                 n_layers: int = 8, n_heads: int = 8):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        # 简化的模型参数
        self.embedding = np.random.randn(vocab_size, d_model).astype(np.float16) * 0.02
        self.lm_head = np.random.randn(d_model, vocab_size).astype(np.float16) * 0.02
        
        # KV Cache
        self.kv_cache = KVCache(n_layers, n_heads, d_model // n_heads)
        
        # 统计
        self.stats = {'requests': 0, 'tokens': 0, 'latency': 0}
    
    def generate(self, input_ids: List[int], config: GenerationConfig = None) -> List[int]:
        """生成回复"""
        if config is None:
            config = GenerationConfig()
        
        self.stats['requests'] += 1
        start_time = time.time()
        
        output = list(input_ids)
        self.kv_cache.reset()
        
        for _ in range(config.max_tokens):
            next_token = self._sample_next_token(output, config)
            output.append(next_token)
            self.stats['tokens'] += 1
            
            if next_token == 2:  # EOS
                break
        
        self.stats['latency'] += time.time() - start_time
        return output[len(input_ids):]
    
    async def generate_stream(self, input_ids: List[int], 
                              config: GenerationConfig = None) -> AsyncGenerator[int, None]:
        """流式生成"""
        if config is None:
            config = GenerationConfig()
        
        output = list(input_ids)
        self.kv_cache.reset()
        
        for _ in range(config.max_tokens):
            next_token = self._sample_next_token(output, config)
            output.append(next_token)
            yield next_token
            
            if next_token == 2:
                break
    
    def _sample_next_token(self, input_ids: List[int], config: GenerationConfig) -> int:
        """采样下一个 token"""
        # 简化的采样
        logits = np.random.randn(self.vocab_size)
        logits = logits / config.temperature
        
        # Top-K
        top_k_indices = np.argsort(logits)[-config.top_k:]
        mask = np.ones(self.vocab_size, dtype=bool)
        mask[top_k_indices] = False
        logits[mask] = -float('inf')
        
        # Top-P
        sorted_indices = np.argsort(logits)[::-1]
        probs = np.exp(logits[sorted_indices])
        probs = probs / probs.sum()
        cumulative = np.cumsum(probs)
        
        top_p_mask = cumulative <= config.top_p
        if not any(top_p_mask):
            top_p_mask[0] = True
        
        candidates = sorted_indices[top_p_mask]
        probs_p = probs[top_p_mask]
        probs_p = probs_p / probs_p.sum()
        
        return int(np.random.choice(candidates, p=probs_p))
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if self.stats['requests'] == 0:
            return self.stats
        
        return {
            **self.stats,
            'avg_latency_ms': self.stats['latency'] / self.stats['requests'] * 1000,
            'throughput': self.stats['tokens'] / max(self.stats['latency'], 0.001)
        }


# ========== 4. 性能优化 ==========

def compare_quantization():
    """对比量化效果"""
    print("量化效果对比:")
    print("-" * 50)
    
    d = 4096
    weight = np.random.randn(d, d).astype(np.float32)
    
    # FP32
    fp32_size = weight.nbytes / (1024 ** 2)
    print(f"FP32: {fp32_size:.2f} MB")
    
    # FP16
    fp16_size = weight.astype(np.float16).nbytes / (1024 ** 2)
    print(f"FP16: {fp16_size:.2f} MB (压缩 {fp32_size/fp16_size:.1f}x)")
    
    # INT8
    quantizer = INT8Quantizer()
    quantized, _ = quantizer.quantize(weight)
    int8_size = quantized.nbytes / (1024 ** 2)
    print(f"INT8: {int8_size:.2f} MB (压缩 {fp32_size/int8_size:.1f}x)")
    
    # INT4 (模拟)
    int4_size = int8_size / 2
    print(f"INT4: ~{int4_size:.2f} MB (压缩 {fp32_size/int4_size:.1f}x)")


def kv_cache_memory():
    """计算 KV Cache 内存"""
    print("\nKV Cache 内存占用:")
    print("-" * 50)
    
    configs = [
        ("LLaMA-7B", 32, 32, 128, 4096),
        ("LLaMA-70B", 80, 64, 128, 4096),
    ]
    
    for name, n_layers, n_heads, d_k, max_seq in configs:
        kv = KVCache(n_layers, n_heads, d_k, max_seq)
        print(f"{name}: {kv.memory_size_mb():.1f} MB (seq_len={max_seq})")


def benchmark_inference():
    """推理基准测试"""
    print("\n推理性能测试:")
    print("-" * 50)
    
    engine = LLMInferenceEngine(vocab_size=1000, d_model=256)
    tokenizer = SimpleTokenizer(1000)
    
    # 测试
    prompts = ["测试输入"] * 5
    config = GenerationConfig(max_tokens=50)
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        output_ids = engine.generate(input_ids, config)
    
    stats = engine.get_stats()
    print(f"请求数: {stats['requests']}")
    print(f"总 token 数: {stats['tokens']}")
    print(f"平均延迟: {stats['avg_latency_ms']:.2f} ms")
    print(f"吞吐量: {stats['throughput']:.2f} tokens/s")


# ========== 主函数 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("模型部署与推理优化示例")
    print("=" * 60)
    
    compare_quantization()
    kv_cache_memory()
    benchmark_inference()
    
    print("\n✅ 部署示例完成!")
