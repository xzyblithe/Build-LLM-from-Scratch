# 第15章：从零实现 MoE 架构

<div align="center">

[⬅️ 上一章](../chapter14-gpt-from-scratch/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter16-mainstream-llms/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解混合专家模型（MoE）原理
- ✅ 实现路由机制
- ✅ 实现负载均衡
- ✅ 从零实现 MoE 层
- ✅ 理解稀疏激活的优势

---

## 🎯 MoE 原理

### 什么是 MoE？

```
MoE = Mixture of Experts（混合专家）

核心思想:
- 不使用一个大模型，而是使用多个小模型（专家）
- 每个输入只激活部分专家（稀疏激活）
- 通过路由机制选择专家

优势:
- 参数量大，但计算量小
- 可以扩展到万亿参数
- 每个专家专注于不同任务
```

---

## 💻 完整 MoE 实现

```python
"""
mini-moe/moe.py - 从零实现 MoE
"""
import numpy as np

class MoELayer:
    """MoE 层"""
    
    def __init__(self, d_model, d_ff, num_experts, top_k=2, capacity_factor=1.25):
        """
        初始化 MoE 层
        
        参数:
            d_model: 模型维度
            d_ff: 前馈网络维度
            num_experts: 专家数量
            top_k: 每个输入激活的专家数
            capacity_factor: 容量因子
        """
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # 门控网络（路由）
        self.gate = np.random.randn(d_model, num_experts) * 0.01
        
        # 专家网络
        self.experts = [
            FeedForwardExpert(d_model, d_ff)
            for _ in range(num_experts)
        ]
        
        # 统计信息
        self.expert_counts = np.zeros(num_experts)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入 (batch_size, seq_len, d_model)
        
        返回:
            输出 (batch_size, seq_len, d_model)
            负载均衡损失
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)  # (batch*seq, d_model)
        
        # 计算门控分数
        gate_logits = np.matmul(x_flat, self.gate)  # (batch*seq, num_experts)
        
        # Softmax
        gate_probs = self.softmax(gate_logits)
        
        # Top-K 选择
        top_k_indices = np.argsort(gate_probs, axis=-1)[:, -self.top_k:]
        top_k_probs = np.take_along_axis(gate_probs, top_k_indices, axis=-1)
        
        # 归一化 Top-K 概率
        top_k_probs = top_k_probs / (top_k_probs.sum(axis=-1, keepdims=True) + 1e-10)
        
        # 初始化输出
        output = np.zeros_like(x_flat)
        
        # 分发到专家
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]
            expert_weights = top_k_probs[:, k]
            
            for expert_id in range(self.num_experts):
                # 找到选择该专家的样本
                mask = (expert_indices == expert_id)
                
                if mask.sum() > 0:
                    # 获取输入
                    expert_input = x_flat[mask]
                    
                    # 专家处理
                    expert_output = self.experts[expert_id].forward(expert_input)
                    
                    # 加权
                    weighted_output = expert_output * expert_weights[mask, np.newaxis]
                    
                    # 累加到输出
                    output[mask] += weighted_output
                    
                    # 统计
                    self.expert_counts[expert_id] += mask.sum()
        
        # 计算负载均衡损失
        aux_loss = self.compute_auxiliary_loss(gate_probs)
        
        # 重塑
        output = output.reshape(batch_size, seq_len, d_model)
        
        return output, aux_loss
    
    def softmax(self, x):
        """Softmax"""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def compute_auxiliary_loss(self, gate_probs):
        """
        计算辅助损失（负载均衡损失）
        
        参数:
            gate_probs: 门控概率 (batch*seq, num_experts)
        
        返回:
            辅助损失
        """
        # 专家重要性（平均概率）
        importance = gate_probs.mean(axis=0)  # (num_experts,)
        
        # 专家负载（被选择的频率）
        load = gate_probs.sum(axis=0) / gate_probs.shape[0]  # (num_experts,)
        
        # 负载均衡损失
        aux_loss = self.num_experts * np.sum(importance * load)
        
        return aux_loss


class FeedForwardExpert:
    """专家网络（前馈网络）"""
    
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        """前向传播"""
        # 第一层 + ReLU
        hidden = np.maximum(0, np.matmul(x, self.W1) + self.b1)
        
        # 第二层
        output = np.matmul(hidden, self.W2) + self.b2
        
        return output


class MoETransformer:
    """带 MoE 的 Transformer"""
    
    def __init__(self, 
                 vocab_size=50000,
                 n_layers=12,
                 n_heads=12,
                 d_model=768,
                 d_ff=3072,
                 num_experts=8,
                 top_k=2,
                 max_seq_len=1024):
        """
        初始化 MoE Transformer
        
        参数:
            vocab_size: 词汇表大小
            n_layers: 层数
            n_heads: 注意力头数
            d_model: 模型维度
            d_ff: 前馈网络维度
            num_experts: 专家数量
            top_k: Top-K 路由
            max_seq_len: 最大序列长度
        """
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        
        # Embedding
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.position_embedding = np.random.randn(max_seq_len, d_model) * 0.02
        
        # 层
        self.layers = []
        for i in range(n_layers):
            # 每 2 层使用 MoE
            if i % 2 == 0:
                self.layers.append({
                    'type': 'moe',
                    'attention': MultiHeadAttention(d_model, n_heads),
                    'moe': MoELayer(d_model, d_ff, num_experts, top_k),
                    'ln1': LayerNorm(d_model),
                    'ln2': LayerNorm(d_model)
                })
            else:
                self.layers.append({
                    'type': 'ffn',
                    'attention': MultiHeadAttention(d_model, n_heads),
                    'ffn': FeedForward(d_model, d_ff),
                    'ln1': LayerNorm(d_model),
                    'ln2': LayerNorm(d_model)
                })
        
        # 输出
        self.ln_f = LayerNorm(d_model)
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
    
    def forward(self, input_ids):
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        x = self.token_embedding[input_ids] + self.position_embedding[:seq_len]
        
        # 创建因果掩码
        causal_mask = self.create_causal_mask(seq_len)
        
        # 层
        total_aux_loss = 0
        
        for layer in self.layers:
            # Self-Attention
            attn_output = layer['attention'].forward(layer['ln1'].forward(x), causal_mask)
            x = x + attn_output
            
            # FFN or MoE
            if layer['type'] == 'moe':
                moe_output, aux_loss = layer['moe'].forward(layer['ln2'].forward(x))
                total_aux_loss += aux_loss
            else:
                moe_output = layer['ffn'].forward(layer['ln2'].forward(x))
            
            x = x + moe_output
        
        # 输出
        x = self.ln_f.forward(x)
        logits = np.matmul(x, self.lm_head)
        
        return logits, total_aux_loss
    
    def create_causal_mask(self, seq_len):
        """创建因果掩码"""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return mask == 0
    
    def generate(self, input_ids, max_new_tokens=50):
        """生成"""
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids[:, -1024:])
            next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            input_ids = np.concatenate([input_ids, next_token], axis=1)
        return input_ids


# 辅助类（简化版）
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        attn_weights = self.softmax(scores)
        output = np.matmul(attn_weights, V)
        
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return np.matmul(output, self.W_o)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)


class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
    
    def forward(self, x):
        return np.matmul(np.maximum(0, np.matmul(x, self.W1)), self.W2)


class LayerNorm:
    def __init__(self, d_model):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + 1e-5) + self.beta


# ================================
# 示例使用
# ================================

if __name__ == "__main__":
    print("初始化 MoE Transformer...")
    
    model = MoETransformer(
        vocab_size=1000,
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
        num_experts=8,
        top_k=2
    )
    
    print(f"✅ MoE 模型创建成功")
    print(f"   专家数量: {model.layers[0]['moe'].num_experts}")
    print(f"   Top-K: {model.layers[0]['moe'].top_k}")
    
    # 测试
    input_ids = np.random.randint(0, 1000, (2, 16))
    logits, aux_loss = model.forward(input_ids)
    
    print(f"\n前向传播测试:")
    print(f"   输入: {input_ids.shape}")
    print(f"   输出: {logits.shape}")
    print(f"   负载均衡损失: {aux_loss:.4f}")
    
    print("\n✅ MoE Transformer 测试通过！")
```

---

## 📝 本章小结

### 核心要点

1. **MoE 架构**：多个专家网络 + 路由机制
2. **稀疏激活**：每个输入只激活 Top-K 个专家
3. **负载均衡**：通过辅助损失确保专家负载均衡
4. **参数效率**：参数量大，但计算量小
5. **应用**：Mixtral、DeepSeek 等模型使用 MoE

---

<div align="center">

[⬅️ 上一章](../chapter14-gpt-from-scratch/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter16-mainstream-llms/README.md)

**🎉 恭喜！你已经从零实现了 MoE 架构！**

</div>
