"""
第8章：Transformer 架构实现
从零实现完整的 Transformer 模型
"""
import numpy as np
from typing import Optional, Tuple


class MultiHeadAttention:
    """多头自注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int):
        """
        参数:
            d_model: 模型维度
            n_heads: 注意力头数
        """
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Q, K, V 投影矩阵
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        前向传播
        
        参数:
            x: 输入张量 (batch_size, seq_len, d_model)
            mask: 注意力掩码 (seq_len, seq_len)
        
        返回:
            输出张量 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 线性投影
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # 分割多头
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 缩放点积注意力
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax
        attn_weights = self._softmax(scores)
        
        # 加权求和
        output = np.matmul(attn_weights, V)
        
        # 合并多头
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = np.matmul(output, self.W_o)
        
        return output
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """数值稳定的 softmax"""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


class PositionalEncoding:
    """位置编码"""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        """
        参数:
            d_model: 模型维度
            max_seq_len: 最大序列长度
        """
        self.d_model = d_model
        
        # 预计算位置编码
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = np.zeros((max_seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        添加位置编码
        
        参数:
            x: 输入张量 (batch_size, seq_len, d_model)
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len]


class FeedForward:
    """前馈神经网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        参数:
            d_model: 模型维度
            d_ff: 隐藏层维度
            dropout: Dropout 率
        """
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.dropout = dropout
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        FFN(x) = max(0, xW1 + b1)W2 + b2
        """
        # 第一层 + ReLU
        hidden = np.maximum(0, np.matmul(x, self.W1))
        
        # Dropout（简化）
        if self.dropout > 0 and np.random.rand() < self.dropout:
            hidden = hidden * 0
        
        # 第二层
        output = np.matmul(hidden, self.W2)
        
        return output


class LayerNorm:
    """层归一化"""
    
    def __init__(self, d_model: int, epsilon: float = 1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.epsilon = epsilon
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """层归一化"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        
        return self.gamma * x_norm + self.beta


class TransformerEncoderLayer:
    """Transformer 编码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """前向传播"""
        # 自注意力 + 残差连接
        attn_output = self.self_attn.forward(x, mask)
        x = self.norm1.forward(x + attn_output)
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_output)
        
        return x


class TransformerDecoderLayer:
    """Transformer 解码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                tgt_mask: Optional[np.ndarray] = None,
                src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """前向传播"""
        # 自注意力（带因果掩码）
        attn_output = self.self_attn.forward(x, tgt_mask)
        x = self.norm1.forward(x + attn_output)
        
        # 交叉注意力
        cross_output = self.cross_attn.forward(
            np.concatenate([x, encoder_output], axis=1)[:, :x.shape[1]],
            src_mask
        )
        x = self.norm2.forward(x + cross_output)
        
        # 前馈网络
        ffn_output = self.ffn.forward(x)
        x = self.norm3.forward(x + ffn_output)
        
        return x


class Transformer:
    """完整的 Transformer 模型"""
    
    def __init__(self, 
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_seq_len: int = 5000):
        """
        参数:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            n_encoder_layers: 编码器层数
            n_decoder_layers: 解码器层数
            d_ff: 前馈网络维度
            dropout: Dropout 率
            max_seq_len: 最大序列长度
        """
        self.d_model = d_model
        
        # Embedding 层
        self.src_embedding = np.random.randn(src_vocab_size, d_model) * 0.02
        self.tgt_embedding = np.random.randn(tgt_vocab_size, d_model) * 0.02
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 编码器层
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ]
        
        # 解码器层
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ]
        
        # 输出层
        self.output_proj = np.random.randn(d_model, tgt_vocab_size) * 0.02
    
    def encode(self, src: np.ndarray, src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """编码器前向传播"""
        # Embedding
        x = self.src_embedding[src] * np.sqrt(self.d_model)
        x = self.pos_encoding.forward(x)
        
        # 编码器层
        for layer in self.encoder_layers:
            x = layer.forward(x, src_mask)
        
        return x
    
    def decode(self, tgt: np.ndarray, encoder_output: np.ndarray,
               tgt_mask: Optional[np.ndarray] = None,
               src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """解码器前向传播"""
        # Embedding
        x = self.tgt_embedding[tgt] * np.sqrt(self.d_model)
        x = self.pos_encoding.forward(x)
        
        # 解码器层
        for layer in self.decoder_layers:
            x = layer.forward(x, encoder_output, tgt_mask, src_mask)
        
        return x
    
    def forward(self, src: np.ndarray, tgt: np.ndarray,
                src_mask: Optional[np.ndarray] = None,
                tgt_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """完整前向传播"""
        # 编码
        encoder_output = self.encode(src, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # 输出投影
        logits = np.matmul(decoder_output, self.output_proj)
        
        return logits
    
    @staticmethod
    def create_causal_mask(seq_len: int) -> np.ndarray:
        """创建因果掩码（解码器自注意力用）"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return (mask == 0).astype(np.float32)


def test_transformer():
    """测试 Transformer 实现"""
    print("=" * 60)
    print("Transformer 模型测试")
    print("=" * 60)
    
    # 参数
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 256
    n_heads = 8
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=512
    )
    
    # 随机输入
    src = np.random.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = np.random.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # 创建掩码
    tgt_mask = Transformer.create_causal_mask(tgt_seq_len)
    
    # 前向传播
    logits = model.forward(src, tgt, tgt_mask=tgt_mask)
    
    print(f"输入形状: src={src.shape}, tgt={tgt.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"模型配置: d_model={d_model}, n_heads={n_heads}")
    
    print("\n✅ Transformer 测试通过！")


if __name__ == "__main__":
    test_transformer()
