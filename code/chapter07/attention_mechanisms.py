"""
第7章：注意力机制实现
包括 Basic Attention、Self-Attention、Multi-Head Attention
"""
import numpy as np
from typing import Optional


class BasicAttention:
    """基础注意力机制"""
    
    def __init__(self):
        pass
    
    def forward(self, query: np.ndarray, keys: np.ndarray, 
                values: np.ndarray, mask: Optional[np.ndarray] = None) -> tuple:
        """
        基础注意力: score = query · keys
        
        参数:
            query: 查询 (d_k,)
            keys: 键 (seq_len, d_k)
            values: 值 (seq_len, d_v)
        
        返回:
            output: 注意力输出 (d_v,)
            attention_weights: 注意力权重 (seq_len,)
        """
        # 计算注意力分数
        scores = np.matmul(keys, query)  # (seq_len,)
        
        # 掩码
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax
        attention_weights = self._softmax(scores)
        
        # 加权求和
        output = np.matmul(attention_weights, values)
        
        return output, attention_weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)


class ScaledDotProductAttention:
    """缩放点积注意力"""
    
    def __init__(self, d_k: int):
        self.d_k = d_k
    
    def forward(self, query: np.ndarray, key: np.ndarray, 
                value: np.ndarray, mask: Optional[np.ndarray] = None) -> tuple:
        """
        缩放点积注意力
        
        参数:
            query: (batch, heads, seq_len_q, d_k)
            key: (batch, heads, seq_len_k, d_k)
            value: (batch, heads, seq_len_v, d_v)
            mask: (seq_len_q, seq_len_k)
        
        返回:
            output: (batch, heads, seq_len_q, d_v)
            attention: (batch, heads, seq_len_q, seq_len_k)
        """
        # 计算分数
        scores = np.matmul(query, key.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # 掩码
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax
        attention = self._softmax(scores)
        
        # 加权求和
        output = np.matmul(attention, value)
        
        return output, attention
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


class SelfAttention:
    """自注意力机制"""
    
    def __init__(self, d_model: int):
        self.d_model = d_model
        
        # Q, K, V 投影
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        
        self.attention = ScaledDotProductAttention(d_model)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        自注意力
        
        参数:
            x: (batch, seq_len, d_model)
        
        返回:
            output: (batch, seq_len, d_model)
        """
        batch_size = x.shape[0]
        
        # 投影
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # 添加 head 维度
        Q = Q[:, np.newaxis, :, :]
        K = K[:, np.newaxis, :, :]
        V = V[:, np.newaxis, :, :]
        
        # 注意力
        output, _ = self.attention.forward(Q, K, V, mask)
        
        # 移除 head 维度
        output = output[:, 0, :, :]
        
        return output


class MultiHeadAttention:
    """多头注意力"""
    
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 投影矩阵
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, query: np.ndarray, key: np.ndarray, 
                value: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        多头注意力
        
        参数:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
        
        返回:
            output: (batch, seq_len_q, d_model)
        """
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        
        # 线性投影
        Q = np.matmul(query, self.W_q)
        K = np.matmul(key, self.W_k)
        V = np.matmul(value, self.W_v)
        
        # 分割多头
        Q = Q.reshape(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 注意力
        output, _ = self.attention.forward(Q, K, V, mask)
        
        # 合并多头
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)
        
        # 输出投影
        output = np.matmul(output, self.W_o)
        
        return output


class CrossAttention:
    """交叉注意力（用于 Encoder-Decoder）"""
    
    def __init__(self, d_model: int, n_heads: int):
        self.attention = MultiHeadAttention(d_model, n_heads)
    
    def forward(self, decoder_hidden: np.ndarray, 
                encoder_output: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        交叉注意力
        
        参数:
            decoder_hidden: 解码器隐藏状态 (batch, dec_seq, d_model)
            encoder_output: 编码器输出 (batch, enc_seq, d_model)
        """
        # Query 来自解码器，Key 和 Value 来自编码器
        return self.attention.forward(decoder_hidden, encoder_output, encoder_output, mask)


class CausalSelfAttention:
    """因果自注意力（用于 GPT 等）"""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.max_seq_len = max_seq_len
        
        # 预计算因果掩码
        self.causal_mask = np.tril(np.ones((max_seq_len, max_seq_len)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        因果自注意力
        
        参数:
            x: (batch, seq_len, d_model)
        """
        seq_len = x.shape[1]
        mask = self.causal_mask[:seq_len, :seq_len]
        
        return self.attention.forward(x, x, x, mask)


def attention_visualization():
    """注意力可视化示例"""
    print("\n注意力权重可视化:")
    print("-" * 50)
    
    # 示例句子
    tokens = ["我", "爱", "自然语言", "处理"]
    seq_len = len(tokens)
    d_model = 8
    
    # 创建自注意力
    attn = SelfAttention(d_model)
    
    # 模拟输入
    x = np.random.randn(1, seq_len, d_model)
    
    # 获取注意力（简化）
    print("句子:", " ".join(tokens))
    print("\n自注意力让每个词关注到其他所有词")
    print("例如 '自然语言' 会关注到 '处理'（语义相关）")


def test_attention():
    """测试注意力机制"""
    print("=" * 60)
    print("注意力机制测试")
    print("=" * 60)
    
    # 参数
    batch_size = 2
    seq_len = 10
    d_model = 64
    n_heads = 4
    
    # 生成测试数据
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # 测试基础注意力
    print("\n【基础注意力】")
    basic_attn = BasicAttention()
    query = np.random.randn(d_model)
    keys = np.random.randn(seq_len, d_model)
    values = np.random.randn(seq_len, d_model)
    output, weights = basic_attn.forward(query, keys, values)
    print(f"查询形状: {query.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重: {weights.shape}, 和为 {weights.sum():.4f}")
    
    # 测试自注意力
    print("\n【自注意力】")
    self_attn = SelfAttention(d_model)
    output = self_attn.forward(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试多头注意力
    print("\n【多头注意力】")
    mha = MultiHeadAttention(d_model, n_heads)
    output = mha.forward(x, x, x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"头数: {n_heads}, 每头维度: {d_model // n_heads}")
    
    # 测试因果注意力
    print("\n【因果自注意力】")
    causal_attn = CausalSelfAttention(d_model, n_heads)
    output = causal_attn.forward(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("特点: 只能看到当前及之前的位置")
    
    attention_visualization()
    
    print("\n✅ 注意力机制测试通过!")


if __name__ == "__main__":
    test_attention()
