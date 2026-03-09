"""
第13章：从零实现 Transformer（详细版）
完整实现 Transformer 的各个组件
"""
import numpy as np
from typing import Optional


class LayerNorm:
    """层归一化"""
    
    def __init__(self, d_model: int, epsilon: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.epsilon = epsilon
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.epsilon) + self.beta


class ScaledDotProductAttention:
    """缩放点积注意力"""
    
    def __init__(self, d_k: int):
        self.d_k = d_k
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: Optional[np.ndarray] = None) -> tuple:
        """
        前向传播
        
        参数:
            Q: 查询 (batch, heads, seq_len, d_k)
            K: 键 (batch, heads, seq_len, d_k)
            V: 值 (batch, heads, seq_len, d_k)
            mask: 掩码
        
        返回:
            output, attention_weights
        """
        # 计算注意力分数
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax
        attention_weights = self._softmax(scores)
        
        # 加权求和
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


class MultiHeadAttention:
    """多头注意力"""
    
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性投影
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """前向传播"""
        batch_size = q.shape[0]
        
        # 线性投影
        Q = np.matmul(q, self.W_q)
        K = np.matmul(k, self.W_k)
        V = np.matmul(v, self.W_v)
        
        # 分割多头
        Q = Q.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 注意力计算
        output, _ = self.attention.forward(Q, K, V, mask)
        
        # 合并多头
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        
        # 输出投影
        output = np.matmul(output, self.W_o)
        
        return output


class PositionWiseFeedForward:
    """位置前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.W_1 = np.random.randn(d_model, d_ff) * 0.02
        self.W_2 = np.random.randn(d_ff, d_model) * 0.02
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播: FFN(x) = max(0, xW_1)W_2"""
        return np.matmul(np.maximum(0, np.matmul(x, self.W_1)), self.W_2)


class PositionalEncoding:
    """位置编码"""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        # 计算位置编码
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = np.zeros((max_seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]


class EncoderLayer:
    """编码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """前向传播"""
        # 自注意力 + 残差 + LayerNorm
        attn_out = self.self_attn.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_out)
        
        # FFN + 残差 + LayerNorm
        ffn_out = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_out)
        
        return x


class DecoderLayer:
    """解码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                tgt_mask: Optional[np.ndarray] = None,
                src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """前向传播"""
        # 自注意力（因果）
        attn_out = self.self_attn.forward(x, x, x, tgt_mask)
        x = self.norm1.forward(x + attn_out)
        
        # 交叉注意力
        cross_out = self.cross_attn.forward(x, encoder_output, encoder_output, src_mask)
        x = self.norm2.forward(x + cross_out)
        
        # FFN
        ffn_out = self.ffn.forward(x)
        x = self.norm3.forward(x + ffn_out)
        
        return x


class TransformerEncoder:
    """Transformer 编码器"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_seq_len: int = 5000):
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.d_model = d_model
        
        self.layers = [
            EncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """前向传播"""
        # Embedding + Positional Encoding
        x = self.embedding[x] * np.sqrt(self.d_model)
        x = self.pos_encoding.forward(x)
        
        # 编码器层
        for layer in self.layers:
            x = layer.forward(x, mask)
        
        return x


class TransformerDecoder:
    """Transformer 解码器"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_seq_len: int = 5000):
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.d_model = d_model
        
        self.layers = [
            DecoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
        
        self.fc_out = np.random.randn(d_model, vocab_size) * 0.02
    
    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                tgt_mask: Optional[np.ndarray] = None,
                src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """前向传播"""
        # Embedding + Positional Encoding
        x = self.embedding[x] * np.sqrt(self.d_model)
        x = self.pos_encoding.forward(x)
        
        # 解码器层
        for layer in self.layers:
            x = layer.forward(x, encoder_output, tgt_mask, src_mask)
        
        # 输出层
        logits = np.matmul(x, self.fc_out)
        
        return logits


class Transformer:
    """完整 Transformer 模型"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, n_heads: int = 8,
                 n_encoder_layers: int = 6, n_decoder_layers: int = 6,
                 d_ff: int = 2048, max_seq_len: int = 5000):
        """
        参数:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            n_encoder_layers: 编码器层数
            n_decoder_layers: 解码器层数
            d_ff: 前馈网络维度
            max_seq_len: 最大序列长度
        """
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_heads, n_encoder_layers, d_ff, max_seq_len
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, n_heads, n_decoder_layers, d_ff, max_seq_len
        )
    
    def forward(self, src: np.ndarray, tgt: np.ndarray,
                src_mask: Optional[np.ndarray] = None,
                tgt_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """前向传播"""
        # 编码
        encoder_output = self.encoder.forward(src, src_mask)
        
        # 解码
        logits = self.decoder.forward(tgt, encoder_output, tgt_mask, src_mask)
        
        return logits
    
    @staticmethod
    def create_causal_mask(seq_len: int) -> np.ndarray:
        """创建因果掩码"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return (mask == 0).astype(np.float32)
    
    def generate(self, src: np.ndarray, max_len: int = 50) -> np.ndarray:
        """自回归生成"""
        batch_size = src.shape[0]
        
        # 编码
        encoder_output = self.encoder.forward(src)
        
        # 初始化输出
        tgt = np.zeros((batch_size, 1), dtype=np.int64)
        
        for _ in range(max_len):
            # 创建因果掩码
            tgt_mask = self.create_causal_mask(tgt.shape[1])
            
            # 解码
            logits = self.decoder.forward(tgt, encoder_output, tgt_mask=tgt_mask)
            
            # 取最后一个位置
            next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            
            # 拼接
            tgt = np.concatenate([tgt, next_token], axis=1)
        
        return tgt


def test_transformer():
    """测试 Transformer"""
    print("=" * 60)
    print("从零实现 Transformer 测试")
    print("=" * 60)
    
    # 参数
    src_vocab = 1000
    tgt_vocab = 1000
    d_model = 128
    n_heads = 4
    n_layers = 2
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_layers,
        n_decoder_layers=n_layers,
        d_ff=256
    )
    
    # 测试数据
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src = np.random.randint(0, src_vocab, (batch_size, src_seq_len))
    tgt = np.random.randint(0, tgt_vocab, (batch_size, tgt_seq_len))
    
    # 创建掩码
    tgt_mask = Transformer.create_causal_mask(tgt_seq_len)
    
    # 前向传播
    logits = model.forward(src, tgt, tgt_mask=tgt_mask)
    
    print(f"源序列: {src.shape}")
    print(f"目标序列: {tgt.shape}")
    print(f"输出 logits: {logits.shape}")
    print(f"模型配置: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    
    print("\n✅ Transformer 测试通过!")


if __name__ == "__main__":
    test_transformer()
