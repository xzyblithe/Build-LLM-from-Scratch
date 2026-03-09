"""
第16章：主流大模型架构实现
LLaMA、RoPE、SwiGLU 等核心组件
"""
import numpy as np
from typing import Optional


class RMSNorm:
    """RMS 归一化（LLaMA 使用）"""
    
    def __init__(self, dim: int, epsilon: float = 1e-6):
        self.weight = np.ones(dim)
        self.epsilon = epsilon
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)
        return self.weight * x / rms


class RotaryPositionalEmbedding:
    """旋转位置编码"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        self.dim = dim
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
        
        # 预计算 cos 和 sin
        t = np.arange(max_seq_len)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        
        self.cos_cached = np.cos(emb)
        self.sin_cached = np.sin(emb)
    
    def forward(self, x: np.ndarray, start_pos: int = 0) -> np.ndarray:
        """应用旋转位置编码"""
        seq_len = x.shape[2]
        
        cos = self.cos_cached[start_pos:start_pos+seq_len]
        sin = self.sin_cached[start_pos:start_pos+seq_len]
        
        # 扩展维度
        cos = cos[np.newaxis, np.newaxis, :, :]  # (1, 1, seq, dim)
        sin = sin[np.newaxis, np.newaxis, :, :]
        
        # 旋转
        x_rot = self._rotate_half(x)
        return x * cos + x_rot * sin
    
    def _rotate_half(self, x: np.ndarray) -> np.ndarray:
        """旋转一半维度"""
        d = x.shape[-1] // 2
        return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


class SwiGLU:
    """SwiGLU 激活函数（LLaMA 使用）"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.gate_proj = np.random.randn(d_model, d_ff) * 0.02
        self.up_proj = np.random.randn(d_model, d_ff) * 0.02
        self.down_proj = np.random.randn(d_ff, d_model) * 0.02
    
    def silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU/Swish 激活"""
        return x * (1 / (1 + np.exp(-x)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """SwiGLU: down(silu(gate) * up)"""
        gate = np.matmul(x, self.gate_proj)
        up = np.matmul(x, self.up_proj)
        
        hidden = self.silu(gate) * up
        return np.matmul(hidden, self.down_proj)


class LlamaAttention:
    """LLaMA 注意力层"""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = None):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads
        
        # QKV 投影
        self.q_proj = np.random.randn(d_model, n_heads * self.d_k) * 0.02
        self.k_proj = np.random.randn(d_model, self.n_kv_heads * self.d_k) * 0.02
        self.v_proj = np.random.randn(d_model, self.n_kv_heads * self.d_k) * 0.02
        self.o_proj = np.random.randn(n_heads * self.d_k, d_model) * 0.02
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.d_k)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """前向传播"""
        batch, seq, _ = x.shape
        
        # QKV
        Q = np.matmul(x, self.q_proj)
        K = np.matmul(x, self.k_proj)
        V = np.matmul(x, self.v_proj)
        
        # 分头
        Q = Q.reshape(batch, seq, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq, self.n_kv_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq, self.n_kv_heads, self.d_k).transpose(0, 2, 1, 3)
        
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
        causal_mask = np.tril(np.ones((seq, seq)))
        scores = np.where(causal_mask == 0, -1e9, scores)
        
        # Softmax
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        
        # 输出
        out = np.matmul(attn, V)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, -1)
        
        return np.matmul(out, self.o_proj)


class LlamaMLP:
    """LLaMA MLP（使用 SwiGLU）"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.swiglu = SwiGLU(d_model, d_ff)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.swiglu.forward(x)


class LlamaDecoderLayer:
    """LLaMA 解码层"""
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int):
        self.attention = LlamaAttention(d_model, n_heads, n_kv_heads)
        self.mlp = LlamaMLP(d_model, d_ff)
        self.input_layernorm = RMSNorm(d_model)
        self.post_attention_layernorm = RMSNorm(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Pre-Norm + Attention
        residual = x
        x = self.input_layernorm.forward(x)
        x = self.attention.forward(x)
        x = residual + x
        
        # Pre-Norm + MLP
        residual = x
        x = self.post_attention_layernorm.forward(x)
        x = self.mlp.forward(x)
        x = residual + x
        
        return x


class LlamaModel:
    """LLaMA 模型"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_kv_heads: int, n_layers: int, d_ff: int, max_seq_len: int = 2048):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding
        self.embed_tokens = np.random.randn(vocab_size, d_model) * 0.02
        
        # Transformer 层
        self.layers = [
            LlamaDecoderLayer(d_model, n_heads, n_kv_heads, d_ff)
            for _ in range(n_layers)
        ]
        
        # 输出
        self.norm = RMSNorm(d_model)
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
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
    
    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 50) -> np.ndarray:
        """生成文本"""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            input_ids = np.concatenate([input_ids, next_token], axis=1)
        return input_ids


def test_llama():
    """测试 LLaMA 模型"""
    print("=" * 60)
    print("LLaMA 模型测试")
    print("=" * 60)
    
    # 小模型配置
    model = LlamaModel(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_kv_heads=2,  # GQA
        n_layers=4,
        d_ff=512
    )
    
    # 测试
    input_ids = np.random.randint(0, 1000, (2, 16))
    logits = model.forward(input_ids)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {logits.shape}")
    print("关键特性: RoPE, RMSNorm, SwiGLU, GQA")
    
    print("\n✅ LLaMA 测试通过!")


if __name__ == "__main__":
    test_llama()
