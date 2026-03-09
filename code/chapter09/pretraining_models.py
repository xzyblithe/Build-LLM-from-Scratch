"""
第9章：预训练语言模型实现
包括 BERT、GPT 预训练方法
"""
import numpy as np
from typing import Optional, List, Tuple


class MaskedLanguageModel:
    """掩码语言模型（MLM）- BERT 预训练任务"""
    
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        
        # MLM 头
        self.mlm_head = np.random.randn(d_model, vocab_size) * 0.02
    
    def create_masked_input(self, input_ids: np.ndarray, 
                            mask_prob: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        创建掩码输入
        
        参数:
            input_ids: 输入 token IDs
            mask_prob: 掩码概率
        
        返回:
            masked_input: 掩码后的输入
            masked_positions: 被掩码的位置
            masked_labels: 掩码标签
        """
        seq_len = len(input_ids)
        masked_input = input_ids.copy()
        
        # 选择掩码位置
        mask_positions = np.random.rand(seq_len) < mask_prob
        masked_positions = np.where(mask_positions)[0]
        masked_labels = input_ids[masked_positions]
        
        # 掩码策略: 80% [MASK], 10% 随机, 10% 不变
        for pos in masked_positions:
            rand = np.random.rand()
            if rand < 0.8:
                masked_input[pos] = 103  # [MASK] token
            elif rand < 0.9:
                masked_input[pos] = np.random.randint(0, self.vocab_size)
            # else: 保持不变
        
        return masked_input, masked_positions, masked_labels
    
    def forward(self, hidden_states: np.ndarray, 
                masked_positions: np.ndarray) -> np.ndarray:
        """
        MLM 前向传播
        
        参数:
            hidden_states: 编码器输出 (batch, seq_len, d_model)
            masked_positions: 掩码位置
        
        返回:
            logits: 掩码位置的预测 (n_masked, vocab_size)
        """
        # 获取掩码位置的隐藏状态
        masked_hidden = hidden_states[:, masked_positions, :]
        
        # 预测
        logits = np.matmul(masked_hidden, self.mlm_head)
        
        return logits.squeeze(0)


class NextSentencePrediction:
    """下一句预测（NSP）- BERT 预训练任务"""
    
    def __init__(self, d_model: int):
        self.nsp_head = np.random.randn(d_model, 2) * 0.02
    
    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        """
        NSP 前向传播
        
        参数:
            cls_hidden: [CLS] token 的隐藏状态 (batch, d_model)
        
        返回:
            logits: 二分类预测 (batch, 2)
        """
        return np.matmul(cls_hidden, self.nsp_head)


class BERTPretraining:
    """BERT 预训练模型"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, 
                 n_layers: int, max_seq_len: int = 512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.segment_embedding = np.random.randn(2, d_model) * 0.02
        self.position_embedding = np.random.randn(max_seq_len, d_model) * 0.02
        
        # Transformer 编码器（简化）
        self.layers = n_layers
        
        # 预训练头
        self.mlm = MaskedLanguageModel(vocab_size, d_model)
        self.nsp = NextSentencePrediction(d_model)
    
    def forward(self, input_ids: np.ndarray, segment_ids: np.ndarray,
                masked_positions: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播
        
        参数:
            input_ids: (batch, seq_len)
            segment_ids: (batch, seq_len)
            masked_positions: 掩码位置
        
        返回:
            mlm_logits: MLM 预测
            nsp_logits: NSP 预测
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        token_emb = self.token_embedding[input_ids]
        segment_emb = self.segment_embedding[segment_ids]
        position_emb = self.position_embedding[:seq_len]
        
        hidden = token_emb + segment_emb + position_emb
        
        # Transformer 层（简化：只做线性变换）
        # 实际应该有多层 Transformer 编码器
        hidden = hidden  # 简化
        
        # MLM 预测
        if masked_positions is not None:
            mlm_logits = self.mlm.forward(hidden, masked_positions)
        else:
            mlm_logits = None
        
        # NSP 预测（使用 [CLS] token）
        cls_hidden = hidden[:, 0, :]
        nsp_logits = self.nsp.forward(cls_hidden)
        
        return mlm_logits, nsp_logits


class CausalLanguageModel:
    """因果语言模型（CLM）- GPT 预训练任务"""
    
    def __init__(self, vocab_size: int, d_model: int):
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        语言模型头
        
        参数:
            hidden_states: (batch, seq_len, d_model)
        
        返回:
            logits: (batch, seq_len, vocab_size)
        """
        return np.matmul(hidden_states, self.lm_head)
    
    def compute_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        计算交叉熵损失
        
        参数:
            logits: (batch, seq_len, vocab_size)
            labels: (batch, seq_len)
        """
        # Softmax
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        
        # 取正确位置的概率
        batch_size, seq_len, vocab_size = logits.shape
        flat_probs = probs.reshape(-1, vocab_size)
        flat_labels = labels.reshape(-1)
        
        # 交叉熵
        log_probs = np.log(flat_probs[np.arange(len(flat_labels)), flat_labels] + 1e-10)
        loss = -np.mean(log_probs)
        
        return loss


class GPTPretraining:
    """GPT 预训练模型"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, max_seq_len: int = 1024):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.position_embedding = np.random.randn(max_seq_len, d_model) * 0.02
        
        # 语言模型头
        self.lm = CausalLanguageModel(vocab_size, d_model)
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        参数:
            input_ids: (batch, seq_len)
        
        返回:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        token_emb = self.token_embedding[input_ids]
        position_emb = self.position_embedding[:seq_len]
        
        hidden = token_emb + position_emb
        
        # Transformer 解码器（简化）
        # 实际应该有多层带因果掩码的 Transformer
        
        # 语言模型头
        logits = self.lm.forward(hidden)
        
        return logits


class DataLoader:
    """预训练数据加载器"""
    
    def __init__(self, corpus: List[str], vocab_size: int, max_seq_len: int):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
    
    def get_batch(self, batch_size: int) -> np.ndarray:
        """获取一个批次"""
        # 简化：随机生成
        return np.random.randint(0, self.vocab_size, (batch_size, self.max_seq_len))


def test_pretraining():
    """测试预训练模型"""
    print("=" * 60)
    print("预训练语言模型测试")
    print("=" * 60)
    
    vocab_size = 1000
    d_model = 256
    n_heads = 4
    n_layers = 4
    batch_size = 2
    seq_len = 32
    
    # ========== 测试 BERT ==========
    print("\n【BERT 预训练】")
    bert = BERTPretraining(vocab_size, d_model, n_heads, n_layers)
    
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    segment_ids = np.zeros((batch_size, seq_len), dtype=np.int32)
    
    # 创建掩码
    mlm = MaskedLanguageModel(vocab_size, d_model)
    masked_input, masked_pos, masked_labels = mlm.create_masked_input(input_ids[0])
    
    mlm_logits, nsp_logits = bert.forward(input_ids, segment_ids)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"NSP 输出: {nsp_logits.shape}")
    print(f"掩码位置数: {len(masked_pos)}")
    
    # ========== 测试 GPT ==========
    print("\n【GPT 预训练】")
    gpt = GPTPretraining(vocab_size, d_model, n_heads, n_layers)
    
    logits = gpt.forward(input_ids)
    
    # 计算损失
    lm = CausalLanguageModel(vocab_size, d_model)
    labels = input_ids.copy()
    loss = lm.compute_loss(logits, labels)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"语言模型损失: {loss:.4f}")
    
    print("\n✅ 预训练模型测试通过!")


if __name__ == "__main__":
    test_pretraining()
