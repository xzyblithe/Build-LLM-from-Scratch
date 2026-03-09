"""
第6章：循环神经网络（RNN）实现
包括 RNN、LSTM、GRU
"""
import numpy as np
from typing import Optional


class SimpleRNN:
    """简单 RNN 实现"""
    
    def __init__(self, input_size: int, hidden_size: int):
        """
        参数:
            input_size: 输入维度
            hidden_size: 隐藏状态维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 初始化权重
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros(hidden_size)
    
    def forward(self, x: np.ndarray, h_prev: Optional[np.ndarray] = None) -> tuple:
        """
        前向传播
        
        参数:
            x: 输入序列 (seq_len, input_size)
            h_prev: 初始隐藏状态
        
        返回:
            outputs: 所有时间步的输出 (seq_len, hidden_size)
            h_final: 最终隐藏状态
        """
        seq_len = x.shape[0]
        
        if h_prev is None:
            h_prev = np.zeros(self.hidden_size)
        
        outputs = []
        h = h_prev
        
        for t in range(seq_len):
            # h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b)
            h = np.tanh(np.matmul(x[t], self.W_xh) + np.matmul(h, self.W_hh) + self.b_h)
            outputs.append(h)
        
        return np.array(outputs), h


class LSTM:
    """LSTM 实现"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入门、遗忘门、输出门、候选记忆
        self.W_i = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.W_f = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.W_o = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.W_c = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        
        self.b_i = np.zeros(hidden_size)
        self.b_f = np.zeros(hidden_size) + 1  # 遗忘门偏置初始化为1
        self.b_o = np.zeros(hidden_size)
        self.b_c = np.zeros(hidden_size)
    
    def forward(self, x: np.ndarray, init_states: Optional[tuple] = None) -> tuple:
        """
        前向传播
        
        参数:
            x: 输入序列 (seq_len, input_size)
            init_states: (h0, c0)
        
        返回:
            outputs: 输出序列 (seq_len, hidden_size)
            (h_final, c_final): 最终状态
        """
        seq_len = x.shape[0]
        
        if init_states is None:
            h = np.zeros(self.hidden_size)
            c = np.zeros(self.hidden_size)
        else:
            h, c = init_states
        
        outputs = []
        
        for t in range(seq_len):
            # 拼接输入和隐藏状态
            combined = np.concatenate([x[t], h])
            
            # 门控
            i = self._sigmoid(np.matmul(combined, self.W_i) + self.b_i)  # 输入门
            f = self._sigmoid(np.matmul(combined, self.W_f) + self.b_f)  # 遗忘门
            o = self._sigmoid(np.matmul(combined, self.W_o) + self.b_o)  # 输出门
            
            # 候选记忆
            c_tilde = np.tanh(np.matmul(combined, self.W_c) + self.b_c)
            
            # 更新记忆和隐藏状态
            c = f * c + i * c_tilde
            h = o * np.tanh(c)
            
            outputs.append(h)
        
        return np.array(outputs), (h, c)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class GRU:
    """GRU 实现"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 更新门、重置门
        self.W_z = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.W_r = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.W_h = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        
        self.b_z = np.zeros(hidden_size)
        self.b_r = np.zeros(hidden_size)
        self.b_h = np.zeros(hidden_size)
    
    def forward(self, x: np.ndarray, h_prev: Optional[np.ndarray] = None) -> tuple:
        """
        前向传播
        
        参数:
            x: 输入序列 (seq_len, input_size)
            h_prev: 初始隐藏状态
        
        返回:
            outputs: 输出序列
            h_final: 最终隐藏状态
        """
        seq_len = x.shape[0]
        
        if h_prev is None:
            h_prev = np.zeros(self.hidden_size)
        
        outputs = []
        h = h_prev
        
        for t in range(seq_len):
            combined = np.concatenate([x[t], h])
            
            # 更新门
            z = self._sigmoid(np.matmul(combined, self.W_z) + self.b_z)
            
            # 重置门
            r = self._sigmoid(np.matmul(combined, self.W_r) + self.b_r)
            
            # 候选隐藏状态
            combined_r = np.concatenate([x[t], r * h])
            h_tilde = np.tanh(np.matmul(combined_r, self.W_h) + self.b_h)
            
            # 最终隐藏状态
            h = (1 - z) * h + z * h_tilde
            
            outputs.append(h)
        
        return np.array(outputs), h
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class BiLSTM:
    """双向 LSTM"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        self.forward_lstm = LSTM(input_size, hidden_size)
        self.backward_lstm = LSTM(input_size, hidden_size)
    
    def forward(self, x: np.ndarray) -> tuple:
        """
        前向传播
        
        参数:
            x: 输入序列 (seq_len, input_size)
        
        返回:
            output: 双向输出 (seq_len, 2*hidden_size)
            states: (h_forward, c_forward, h_backward, c_backward)
        """
        # 前向
        forward_out, (h_f, c_f) = self.forward_lstm.forward(x)
        
        # 后向
        backward_out, (h_b, c_b) = self.backward_lstm.forward(x[::-1])
        backward_out = backward_out[::-1]  # 反转回来
        
        # 拼接
        output = np.concatenate([forward_out, backward_out], axis=-1)
        
        return output, (h_f, c_f, h_b, c_b)


class RNNLanguageModel:
    """RNN 语言模型"""
    
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int,
                 rnn_type: str = 'lstm'):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        # Embedding
        self.embedding = np.random.randn(vocab_size, embed_size) * 0.01
        
        # RNN
        if rnn_type == 'lstm':
            self.rnn = LSTM(embed_size, hidden_size)
        elif rnn_type == 'gru':
            self.rnn = GRU(embed_size, hidden_size)
        else:
            self.rnn = SimpleRNN(embed_size, hidden_size)
        
        # 输出层
        self.fc = np.random.randn(hidden_size, vocab_size) * 0.01
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        参数:
            input_ids: 输入 token IDs (seq_len,)
        
        返回:
            logits: 输出概率 (seq_len, vocab_size)
        """
        # Embedding
        x = self.embedding[input_ids]
        
        # RNN
        outputs, _ = self.rnn.forward(x)
        
        # 输出
        logits = np.matmul(outputs, self.fc)
        
        return logits
    
    def generate(self, start_token: int, max_length: int = 50) -> list:
        """生成文本"""
        tokens = [start_token]
        
        for _ in range(max_length):
            logits = self.forward(np.array(tokens))
            next_token = np.argmax(logits[-1])
            tokens.append(int(next_token))
            
            if next_token == 2:  # EOS
                break
        
        return tokens


def test_rnn():
    """测试 RNN 模型"""
    print("=" * 60)
    print("RNN 模型测试")
    print("=" * 60)
    
    # 参数
    input_size = 64
    hidden_size = 128
    seq_len = 20
    batch_size = 1
    
    # 生成测试数据
    x = np.random.randn(seq_len, input_size)
    
    # 测试 SimpleRNN
    print("\n【SimpleRNN】")
    rnn = SimpleRNN(input_size, hidden_size)
    outputs, h_final = rnn.forward(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {outputs.shape}")
    print(f"最终隐藏状态: {h_final.shape}")
    
    # 测试 LSTM
    print("\n【LSTM】")
    lstm = LSTM(input_size, hidden_size)
    outputs, (h, c) = lstm.forward(x)
    print(f"输出形状: {outputs.shape}")
    print(f"隐藏状态: {h.shape}, 记忆状态: {c.shape}")
    
    # 测试 GRU
    print("\n【GRU】")
    gru = GRU(input_size, hidden_size)
    outputs, h_final = gru.forward(x)
    print(f"输出形状: {outputs.shape}")
    print(f"最终隐藏状态: {h_final.shape}")
    
    # 测试语言模型
    print("\n【RNN 语言模型】")
    lm = RNNLanguageModel(vocab_size=1000, embed_size=64, hidden_size=128, rnn_type='lstm')
    input_ids = np.random.randint(0, 1000, 10)
    logits = lm.forward(input_ids)
    print(f"输入 IDs: {input_ids.shape}")
    print(f"输出 logits: {logits.shape}")
    
    print("\n✅ RNN 测试通过!")


if __name__ == "__main__":
    test_rnn()
