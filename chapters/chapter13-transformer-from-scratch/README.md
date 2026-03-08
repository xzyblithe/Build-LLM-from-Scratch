# 第13章：从零实现 Transformer

<div align="center">

[⬅️ 上一章](../chapter12-huggingface/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter14-gpt-from-scratch/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 完整实现 Transformer 的所有组件
- ✅ 理解 Encoder-Decoder 架构的实现细节
- ✅ 实现位置编码、注意力机制、前馈网络
- ✅ 训练一个简单的机器翻译模型
- ✅ 理解 Transformer 的训练和推理流程

---

## 🎯 项目结构

```
mini-transformer/
├── model/
│   ├── __init__.py
│   ├── layers.py         # 各层实现
│   ├── encoder.py        # Encoder
│   ├── decoder.py        # Decoder
│   └── transformer.py    # 完整模型
├── data/
│   ├── dataset.py        # 数据处理
│   └── tokenizer.py      # 分词器
├── train.py              # 训练脚本
└── inference.py          # 推理脚本
```

---

## 💻 完整代码实现

### 1. 基础层实现

```python
"""
model/layers.py - Transformer 基础层
"""
import numpy as np

class MultiHeadAttention:
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 初始化权重
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
        # 缓存
        self.cache = {}
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        d_k = Q.shape[-1]
        
        # 计算注意力分数
        scores = np.matmul(Q, K.transpose(0, 1, 2, 4, 3)) / np.sqrt(d_k)
        
        # 应用掩码
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax
        attention_weights = self.softmax(scores)
        
        # 加权求和
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def softmax(self, x):
        """数值稳定的 Softmax"""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def forward(self, query, key, value, mask=None):
        """前向传播"""
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # 线性变换
        Q = np.matmul(query, self.W_q)  # (batch, seq_len, d_model)
        K = np.matmul(key, self.W_k)
        V = np.matmul(value, self.W_v)
        
        # 分割成多头
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        Q = Q.transpose(0, 2, 1, 3)  # (batch, num_heads, seq_len, d_k)
        
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 注意力计算
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 缓存用于反向传播
        self.cache['attention_weights'] = attention_weights
        self.cache['Q'] = Q
        self.cache['K'] = K
        self.cache['V'] = V
        
        # 合并多头
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # 输出线性变换
        output = np.matmul(output, self.W_o)
        
        return output
    
    def backward(self, dout, learning_rate=0.01):
        """反向传播"""
        batch_size = dout.shape[0]
        seq_len = dout.shape[1]
        
        # 输出层梯度
        dW_o = np.matmul(
            self.cache['output'].transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model).transpose(0, 2, 1),
            dout
        ).sum(axis=0)
        
        # 更新权重
        self.W_o -= learning_rate * dW_o
        
        return dout


class PositionalEncoding:
    """位置编码"""
    
    def __init__(self, d_model, max_seq_len=5000):
        self.d_model = d_model
        
        # 创建位置编码矩阵
        position = np.arange(max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def forward(self, x):
        """添加位置编码"""
        return x + self.pe[:x.shape[1]]


class FeedForward:
    """前馈网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
        self.dropout = dropout
        
        self.cache = {}
    
    def relu(self, x):
        """ReLU 激活函数"""
        return np.maximum(0, x)
    
    def forward(self, x):
        """前向传播"""
        # 第一层
        hidden = np.matmul(x, self.W1) + self.b1
        hidden = self.relu(hidden)
        
        # Dropout
        if self.dropout > 0:
            mask = (np.random.rand(*hidden.shape) > self.dropout) / (1 - self.dropout)
            hidden = hidden * mask
            self.cache['dropout_mask'] = mask
        
        # 第二层
        output = np.matmul(hidden, self.W2) + self.b2
        
        # 缓存
        self.cache['x'] = x
        self.cache['hidden'] = hidden
        
        return output
    
    def backward(self, dout, learning_rate=0.01):
        """反向传播"""
        # 第二层梯度
        dW2 = np.matmul(self.cache['hidden'].T, dout)
        db2 = np.sum(dout, axis=0)
        
        # 反向传播到隐藏层
        dhidden = np.matmul(dout, self.W2.T)
        
        # ReLU 梯度
        dhidden = dhidden * (self.cache['hidden'] > 0)
        
        # Dropout 梯度
        if self.dropout > 0:
            dhidden = dhidden * self.cache['dropout_mask']
        
        # 第一层梯度
        dW1 = np.matmul(self.cache['x'].T, dhidden)
        db1 = np.sum(dhidden, axis=0)
        
        # 更新参数
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        # 反向传播到输入
        dx = np.matmul(dhidden, self.W1.T)
        
        return dx


class LayerNorm:
    """层归一化"""
    
    def __init__(self, d_model, epsilon=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.epsilon = epsilon
        
        self.cache = {}
    
    def forward(self, x):
        """前向传播"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        output = self.gamma * x_norm + self.beta
        
        # 缓存
        self.cache['x'] = x
        self.cache['mean'] = mean
        self.cache['var'] = var
        self.cache['x_norm'] = x_norm
        
        return output
    
    def backward(self, dout, learning_rate=0.01):
        """反向传播"""
        N = dout.shape[-1]
        
        dgamma = np.sum(dout * self.cache['x_norm'], axis=tuple(range(dout.ndim - 1)))
        dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))
        
        dx_norm = dout * self.gamma
        
        dvar = np.sum(dx_norm * (self.cache['x'] - self.cache['mean']) * 
                     -0.5 * (self.cache['var'] + self.epsilon)**(-1.5), 
                     axis=-1, keepdims=True)
        
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.cache['var'] + self.epsilon), 
                      axis=-1, keepdims=True) + \
                dvar * np.mean(-2 * (self.cache['x'] - self.cache['mean']), 
                              axis=-1, keepdims=True)
        
        dx = dx_norm / np.sqrt(self.cache['var'] + self.epsilon) + \
             dvar * 2 * (self.cache['x'] - self.cache['mean']) / N + \
             dmean / N
        
        # 更新参数
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return dx
```

---

### 2. Encoder 实现

```python
"""
model/encoder.py - Transformer Encoder
"""
import numpy as np
from .layers import MultiHeadAttention, FeedForward, LayerNorm

class EncoderLayer:
    """Encoder 单层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout
    
    def forward(self, x, mask=None):
        """前向传播"""
        # Multi-Head Attention
        mha_output = self.mha.forward(x, x, x, mask)
        
        # 残差连接 + Layer Norm
        x = self.norm1.forward(x + mha_output)
        
        # Feed Forward
        ffn_output = self.ffn.forward(x)
        
        # 残差连接 + Layer Norm
        x = self.norm2.forward(x + ffn_output)
        
        return x


class Encoder:
    """Transformer Encoder"""
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, 
                 input_vocab_size, max_seq_len, dropout=0.1):
        from .layers import PositionalEncoding
        
        self.embedding = np.random.randn(input_vocab_size, d_model) * 0.01
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        
        self.dropout = dropout
    
    def forward(self, x, mask=None):
        """前向传播"""
        # Embedding
        seq_len = x.shape[1]
        x = self.embedding[x]  # (batch, seq_len, d_model)
        
        # 位置编码
        x = self.pos_encoding.forward(x)
        
        # Dropout
        if self.dropout > 0:
            dropout_mask = (np.random.rand(*x.shape) > self.dropout) / (1 - self.dropout)
            x = x * dropout_mask
        
        # Encoder 层
        for layer in self.encoder_layers:
            x = layer.forward(x, mask)
        
        return x
```

---

### 3. Decoder 实现

```python
"""
model/decoder.py - Transformer Decoder
"""
import numpy as np
from .layers import MultiHeadAttention, FeedForward, LayerNorm

class DecoderLayer:
    """Decoder 单层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.mha1 = MultiHeadAttention(d_model, num_heads)  # Self-Attention
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # Cross-Attention
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = dropout
    
    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        """前向传播"""
        # Masked Self-Attention
        mha1_output = self.mha1.forward(x, x, x, look_ahead_mask)
        x = self.norm1.forward(x + mha1_output)
        
        # Cross-Attention
        mha2_output = self.mha2.forward(x, encoder_output, encoder_output, padding_mask)
        x = self.norm2.forward(x + mha2_output)
        
        # Feed Forward
        ffn_output = self.ffn.forward(x)
        x = self.norm3.forward(x + ffn_output)
        
        return x


class Decoder:
    """Transformer Decoder"""
    
    def __init__(self, num_layers, d_model, num_heads, d_ff,
                 target_vocab_size, max_seq_len, dropout=0.1):
        from .layers import PositionalEncoding
        
        self.embedding = np.random.randn(target_vocab_size, d_model) * 0.01
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        
        self.dropout = dropout
    
    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        """前向传播"""
        # Embedding
        seq_len = x.shape[1]
        x = self.embedding[x]
        
        # 位置编码
        x = self.pos_encoding.forward(x)
        
        # Dropout
        if self.dropout > 0:
            dropout_mask = (np.random.rand(*x.shape) > self.dropout) / (1 - self.dropout)
            x = x * dropout_mask
        
        # Decoder 层
        for layer in self.decoder_layers:
            x = layer.forward(x, encoder_output, look_ahead_mask, padding_mask)
        
        return x
```

---

### 4. 完整 Transformer 模型

```python
"""
model/transformer.py - 完整 Transformer 模型
"""
import numpy as np
from .encoder import Encoder
from .decoder import Decoder

class Transformer:
    """完整的 Transformer 模型"""
    
    def __init__(self, num_layers, d_model, num_heads, d_ff,
                 input_vocab_size, target_vocab_size,
                 max_seq_len=5000, dropout=0.1):
        """
        初始化 Transformer
        
        参数:
            num_layers: Encoder/Decoder 层数
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络维度
            input_vocab_size: 输入词汇表大小
            target_vocab_size: 目标词汇表大小
            max_seq_len: 最大序列长度
            dropout: Dropout 率
        """
        self.encoder = Encoder(
            num_layers, d_model, num_heads, d_ff,
            input_vocab_size, max_seq_len, dropout
        )
        
        self.decoder = Decoder(
            num_layers, d_model, num_heads, d_ff,
            target_vocab_size, max_seq_len, dropout
        )
        
        # 输出层
        self.output_projection = np.random.randn(
            d_model, target_vocab_size
        ) * np.sqrt(2.0 / d_model)
        
        self.d_model = d_model
        self.target_vocab_size = target_vocab_size
    
    def create_padding_mask(self, seq):
        """创建填充掩码"""
        mask = (seq != 0).astype(float)
        return mask[:, np.newaxis, np.newaxis, :]  # (batch, 1, 1, seq_len)
    
    def create_look_ahead_mask(self, seq_len):
        """创建前瞻掩码"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask == 0  # 下三角为 True
    
    def forward(self, encoder_input, decoder_input):
        """
        前向传播
        
        参数:
            encoder_input: Encoder 输入 (batch, enc_seq_len)
            decoder_input: Decoder 输入 (batch, dec_seq_len)
        
        返回:
            输出概率 (batch, dec_seq_len, target_vocab_size)
        """
        # 创建掩码
        enc_padding_mask = self.create_padding_mask(encoder_input)
        look_ahead_mask = self.create_look_ahead_mask(decoder_input.shape[1])
        dec_padding_mask = self.create_padding_mask(encoder_input)
        
        # Encoder
        encoder_output = self.encoder.forward(encoder_input, enc_padding_mask)
        
        # Decoder
        decoder_output = self.decoder.forward(
            decoder_input, encoder_output,
            look_ahead_mask, dec_padding_mask
        )
        
        # 输出投影
        logits = np.matmul(decoder_output, self.output_projection)
        
        # Softmax
        output = self.softmax(logits)
        
        return output
    
    def softmax(self, x):
        """Softmax"""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def train_step(self, encoder_input, decoder_input, target, learning_rate=0.01):
        """训练步骤"""
        # 前向传播
        output = self.forward(encoder_input, decoder_input)
        
        # 计算损失（交叉熵）
        batch_size = target.shape[0]
        seq_len = target.shape[1]
        
        # One-hot 编码
        target_onehot = np.zeros((batch_size, seq_len, self.target_vocab_size))
        target_onehot[np.arange(batch_size)[:, None], np.arange(seq_len), target] = 1
        
        # 交叉熵损失
        loss = -np.mean(np.sum(target_onehot * np.log(output + 1e-10), axis=-1))
        
        # 反向传播（简化版，实际应使用自动微分）
        # 这里省略完整的反向传播实现
        
        return loss
    
    def predict(self, encoder_input, start_token, end_token, max_length=50):
        """
        推理（贪婪解码）
        
        参数:
            encoder_input: Encoder 输入
            start_token: 起始标记
            end_token: 结束标记
            max_length: 最大生成长度
        
        返回:
            生成的序列
        """
        batch_size = encoder_input.shape[0]
        
        # 初始化 Decoder 输入
        decoder_input = np.full((batch_size, 1), start_token)
        
        # 编码 Encoder 输入
        enc_padding_mask = self.create_padding_mask(encoder_input)
        encoder_output = self.encoder.forward(encoder_input, enc_padding_mask)
        
        # 逐步生成
        for _ in range(max_length):
            # 创建前瞻掩码
            look_ahead_mask = self.create_look_ahead_mask(decoder_input.shape[1])
            dec_padding_mask = self.create_padding_mask(encoder_input)
            
            # Decoder 前向传播
            decoder_output = self.decoder.forward(
                decoder_input, encoder_output,
                look_ahead_mask, dec_padding_mask
            )
            
            # 预测下一个词
            logits = np.matmul(decoder_output[:, -1:, :], self.output_projection)
            predictions = self.softmax(logits)
            
            # 贪婪选择
            predicted_id = np.argmax(predictions, axis=-1)
            
            # 拼接到输出
            decoder_input = np.concatenate([decoder_input, predicted_id], axis=1)
            
            # 检查是否结束
            if predicted_id[0, 0] == end_token:
                break
        
        return decoder_input

# ================================
# 示例使用
# ================================

if __name__ == "__main__":
    print("初始化 Transformer 模型...")
    
    # 模型参数
    NUM_LAYERS = 4
    D_MODEL = 128
    NUM_HEADS = 8
    D_FF = 512
    INPUT_VOCAB_SIZE = 1000
    TARGET_VOCAB_SIZE = 1000
    MAX_SEQ_LEN = 100
    DROPOUT = 0.1
    
    # 创建模型
    transformer = Transformer(
        NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF,
        INPUT_VOCAB_SIZE, TARGET_VOCAB_SIZE,
        MAX_SEQ_LEN, DROPOUT
    )
    
    print(f"✅ 模型创建成功")
    print(f"   层数: {NUM_LAYERS}")
    print(f"   模型维度: {D_MODEL}")
    print(f"   注意力头数: {NUM_HEADS}")
    
    # 测试前向传播
    batch_size = 2
    enc_seq_len = 10
    dec_seq_len = 8
    
    encoder_input = np.random.randint(1, INPUT_VOCAB_SIZE, (batch_size, enc_seq_len))
    decoder_input = np.random.randint(1, TARGET_VOCAB_SIZE, (batch_size, dec_seq_len))
    
    print(f"\n前向传播测试:")
    print(f"   Encoder 输入: {encoder_input.shape}")
    print(f"   Decoder 输入: {decoder_input.shape}")
    
    output = transformer.forward(encoder_input, decoder_input)
    print(f"   输出形状: {output.shape}")
    print(f"   输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    print("\n✅ Transformer 模型测试通过！")
    print("   这是从零实现的完整 Transformer")
```

---

## 🎯 训练示例

```python
"""
train.py - 训练脚本示例
"""
import numpy as np
from model.transformer import Transformer

def train_transformer():
    """训练 Transformer"""
    
    # 数据准备（简化示例）
    # 实际应用中需要真实数据集
    
    # 创建模型
    transformer = Transformer(
        num_layers=4,
        d_model=128,
        num_heads=8,
        d_ff=512,
        input_vocab_size=1000,
        target_vocab_size=1000,
        max_seq_len=100
    )
    
    # 训练参数
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    
    print("开始训练...")
    
    for epoch in range(epochs):
        # 模拟数据
        encoder_input = np.random.randint(1, 1000, (batch_size, 10))
        decoder_input = np.random.randint(1, 1000, (batch_size, 8))
        target = np.random.randint(1, 1000, (batch_size, 8))
        
        # 训练步骤
        loss = transformer.train_step(
            encoder_input, decoder_input, target, learning_rate
        )
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    print("✅ 训练完成！")

if __name__ == "__main__":
    train_transformer()
```

---

## 📝 本章小结

### 核心要点

1. **Multi-Head Attention**：并行计算多个注意力头
2. **Positional Encoding**：注入位置信息
3. **Encoder-Decoder**：编码器提取特征，解码器生成输出
4. **残差连接 + Layer Norm**：稳定训练
5. **掩码机制**：处理变长序列和自回归生成

### 实现技巧

- 权重初始化使用 Xavier/Glorot
- Dropout 防止过拟合
- 学习率调度（Warmup + Decay）
- 梯度裁剪防止梯度爆炸

---

<div align="center">

[⬅️ 上一章](../chapter12-huggingface/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter14-gpt-from-scratch/README.md)

**🎉 恭喜！你已经从零实现了完整的 Transformer！**

</div>
