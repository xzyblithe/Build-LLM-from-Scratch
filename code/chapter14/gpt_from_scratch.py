"""
第14章：从零实现 GPT
完整实现 GPT 模型的各个组件
"""
import numpy as np
from typing import Optional, List


class LayerNorm:
    """层归一化"""
    
    def __init__(self, n_embd: int, epsilon: float = 1e-5):
        self.gamma = np.ones(n_embd)
        self.beta = np.zeros(n_embd)
        self.epsilon = epsilon
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.epsilon) + self.beta


class GPTAttention:
    """GPT 注意力层（带因果掩码）"""
    
    def __init__(self, n_embd: int, n_heads: int, block_size: int):
        assert n_embd % n_heads == 0
        
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.d_k = n_embd // n_heads
        self.block_size = block_size
        
        # 投影层
        self.c_attn = np.random.randn(n_embd, 3 * n_embd) * 0.02  # QKV 合并
        self.c_proj = np.random.randn(n_embd, n_embd) * 0.02
        
        # 因果掩码
        self.causal_mask = np.tril(np.ones((block_size, block_size)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        参数:
            x: (batch, seq_len, n_embd)
        """
        B, T, C = x.shape
        
        # QKV 投影
        qkv = np.matmul(x, self.c_attn)
        q, k, v = np.split(qkv, 3, axis=-1)
        
        # 分头
        q = q.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 注意力分数
        att = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # 因果掩码
        att = np.where(self.causal_mask[:T, :T] == 0, -1e10, att)
        
        # Softmax
        att = np.exp(att - np.max(att, axis=-1, keepdims=True))
        att = att / np.sum(att, axis=-1, keepdims=True)
        
        # 加权求和
        y = np.matmul(att, v)
        
        # 合并多头
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # 输出投影
        y = np.matmul(y, self.c_proj)
        
        return y


class GPTMLP:
    """GPT 前馈网络"""
    
    def __init__(self, n_embd: int):
        self.c_fc = np.random.randn(n_embd, 4 * n_embd) * 0.02
        self.c_proj = np.random.randn(4 * n_embd, n_embd) * 0.02
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU 激活函数"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # 升维 + GELU + 降维
        x = np.matmul(x, self.c_fc)
        x = self.gelu(x)
        x = np.matmul(x, self.c_proj)
        return x


class GPTBlock:
    """GPT Transformer 块"""
    
    def __init__(self, n_embd: int, n_heads: int, block_size: int):
        self.ln_1 = LayerNorm(n_embd)
        self.attn = GPTAttention(n_embd, n_heads, block_size)
        self.ln_2 = LayerNorm(n_embd)
        self.mlp = GPTMLP(n_embd)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播（Pre-Norm 结构）"""
        # 注意力 + 残差
        x = x + self.attn.forward(self.ln_1.forward(x))
        
        # MLP + 残差
        x = x + self.mlp.forward(self.ln_2.forward(x))
        
        return x


class GPT:
    """GPT 语言模型"""
    
    def __init__(self, vocab_size: int, n_embd: int, n_heads: int,
                 n_layers: int, block_size: int):
        """
        参数:
            vocab_size: 词汇表大小
            n_embd: 嵌入维度
            n_heads: 注意力头数
            n_layers: 层数
            block_size: 最大序列长度
        """
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layers = n_layers
        self.block_size = block_size
        
        # Token Embedding
        self.wte = np.random.randn(vocab_size, n_embd) * 0.02
        
        # Position Embedding
        self.wpe = np.random.randn(block_size, n_embd) * 0.02
        
        # Transformer 块
        self.blocks = [
            GPTBlock(n_embd, n_heads, block_size)
            for _ in range(n_layers)
        ]
        
        # 最终层归一化
        self.ln_f = LayerNorm(n_embd)
        
        # 语言模型头
        self.lm_head = self.wte  # 权重共享
    
    def forward(self, idx: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        参数:
            idx: 输入 token IDs (batch, seq_len)
        
        返回:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = idx.shape
        
        # Token + Position Embedding
        tok_emb = self.wte[idx]  # (B, T, n_embd)
        pos_emb = self.wpe[:T]   # (T, n_embd)
        x = tok_emb + pos_emb
        
        # Transformer 块
        for block in self.blocks:
            x = block.forward(x)
        
        # 最终归一化
        x = self.ln_f.forward(x)
        
        # 语言模型头
        logits = np.matmul(x, self.lm_head.T)
        
        return logits
    
    def generate(self, idx: np.ndarray, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None) -> np.ndarray:
        """
        文本生成
        
        参数:
            idx: 输入 token IDs (batch, seq_len)
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_k: Top-K 采样
        """
        for _ in range(max_new_tokens):
            # 截断到 block_size
            idx_cond = idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size:]
            
            # 前向传播
            logits = self.forward(idx_cond)
            
            # 取最后一个位置
            logits = logits[:, -1, :] / temperature
            
            # Top-K 过滤
            if top_k is not None:
                v = np.sort(logits, axis=-1)[:, -top_k]
                logits = np.where(logits < v[:, np.newaxis], -float('inf'), logits)
            
            # Softmax
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            # 采样
            next_token = np.array([
                np.random.choice(self.vocab_size, p=p) for p in probs
            ])[:, np.newaxis]
            
            # 拼接
            idx = np.concatenate([idx, next_token], axis=1)
        
        return idx


class GPTTokenizer:
    """简单的字符级分词器"""
    
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
    
    def train(self, text: str):
        """从文本构建词表"""
        chars = sorted(set(text))
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
    
    def encode(self, text: str) -> List[int]:
        """编码"""
        return [self.char2idx[c] for c in text if c in self.char2idx]
    
    def decode(self, tokens: List[int]) -> str:
        """解码"""
        return ''.join(self.idx2char.get(t, '') for t in tokens)


def test_gpt():
    """测试 GPT 模型"""
    print("=" * 60)
    print("GPT 模型测试")
    print("=" * 60)
    
    # 参数
    vocab_size = 65  # 字符级
    n_embd = 128
    n_heads = 4
    n_layers = 4
    block_size = 64
    
    # 创建模型
    model = GPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_heads=n_heads,
        n_layers=n_layers,
        block_size=block_size
    )
    
    # 测试前向传播
    batch_size = 2
    seq_len = 16
    
    idx = np.random.randint(0, vocab_size, (batch_size, seq_len))
    logits = model.forward(idx)
    
    print(f"输入形状: {idx.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"模型配置:")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - n_embd: {n_embd}")
    print(f"  - n_heads: {n_heads}")
    print(f"  - n_layers: {n_layers}")
    print(f"  - block_size: {block_size}")
    
    # 测试生成
    print("\n测试文本生成...")
    start_tokens = np.array([[0, 1, 2]])  # 开始 token
    generated = model.generate(start_tokens, max_new_tokens=20, temperature=0.8)
    print(f"生成长度: {generated.shape}")
    
    print("\n✅ GPT 模型测试通过!")


def estimate_parameters(model: GPT) -> int:
    """估算模型参数量"""
    # Token embedding
    params = model.vocab_size * model.n_embd
    
    # Position embedding
    params += model.block_size * model.n_embd
    
    # Transformer blocks
    for block in model.blocks:
        # Attention: c_attn (n_embd -> 3*n_embd) + c_proj
        params += model.n_embd * 3 * model.n_embd
        params += model.n_embd * model.n_embd
        
        # MLP: c_fc (n_embd -> 4*n_embd) + c_proj
        params += model.n_embd * 4 * model.n_embd
        params += 4 * model.n_embd * model.n_embd
        
        # LayerNorm
        params += 2 * model.n_embd * 2  # gamma + beta for each ln
    
    # Final LayerNorm
    params += model.n_embd * 2
    
    return params


if __name__ == "__main__":
    test_gpt()
    
    # 参数量估算
    model = GPT(vocab_size=50000, n_embd=768, n_heads=12, n_layers=12, block_size=1024)
    params = estimate_parameters(model)
    print(f"\nGPT-2 Small 参数量估算: {params:,} ({params/1e6:.1f}M)")
