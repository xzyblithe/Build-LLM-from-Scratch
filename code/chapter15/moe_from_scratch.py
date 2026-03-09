"""
第15章：从零实现 MoE（混合专家模型）
"""
import numpy as np
from typing import List, Tuple


class Expert:
    """单个专家网络"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        hidden = np.maximum(0, np.matmul(x, self.W1))  # ReLU
        return np.matmul(hidden, self.W2)


class Router:
    """路由网络（Top-K 门控）"""
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        self.W = np.random.randn(d_model, num_experts) * 0.01
        self.num_experts = num_experts
        self.top_k = top_k
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播
        
        返回:
            gates: 门控权重 (batch, seq, top_k)
            indices: 专家索引 (batch, seq, top_k)
        """
        # 计算路由分数
        logits = np.matmul(x, self.W)  # (batch, seq, num_experts)
        
        # Top-K 选择
        top_k_indices = np.argsort(logits, axis=-1)[..., -self.top_k:]
        top_k_logits = np.take_along_axis(logits, top_k_indices, axis=-1)
        
        # Softmax
        gates = np.exp(top_k_logits - np.max(top_k_logits, axis=-1, keepdims=True))
        gates = gates / np.sum(gates, axis=-1, keepdims=True)
        
        return gates, top_k_indices


class MoELayer:
    """MoE 层"""
    
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2):
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.experts = [Expert(d_model, d_ff) for _ in range(num_experts)]
        self.router = Router(d_model, num_experts, top_k)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        前向传播
        
        返回:
            output: 输出张量
            load_balance_loss: 负载均衡损失
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)  # (batch*seq, d_model)
        
        # 路由
        gates, indices = self.router.forward(x)
        gates_flat = gates.reshape(-1, self.top_k)
        indices_flat = indices.reshape(-1, self.top_k)
        
        # 初始化输出
        output = np.zeros_like(x_flat)
        
        # 专家计算（简化版，实际应该批量处理）
        for i in range(self.num_experts):
            # 找到路由到专家 i 的位置
            mask = (indices_flat == i)
            if not np.any(mask):
                continue
            
            # 获取对应的输入和权重
            expert_input = x_flat[np.any(mask, axis=1)]
            expert_gates = gates_flat[mask]
            
            # 专家计算
            expert_output = self.experts[i].forward(expert_input)
            
            # 加权求和
            # 简化处理
            pass
        
        # 简化实现：使用均匀路由
        for i in range(batch_size * seq_len):
            for k in range(self.top_k):
                expert_idx = indices_flat[i, k]
                gate = gates_flat[i, k]
                output[i] += gate * self.experts[expert_idx].forward(x_flat[i:i+1])[0]
        
        # 负载均衡损失（简化）
        load_balance_loss = self._compute_load_balance_loss(gates_flat, indices_flat)
        
        return output.reshape(batch_size, seq_len, d_model), load_balance_loss
    
    def _compute_load_balance_loss(self, gates: np.ndarray, indices: np.ndarray) -> float:
        """计算负载均衡损失"""
        # 专家选择频率
        expert_counts = np.zeros(self.num_experts)
        for i in range(self.num_experts):
            expert_counts[i] = np.sum(indices == i)
        
        # 归一化
        total = np.sum(expert_counts)
        if total > 0:
            expert_freq = expert_counts / total
        else:
            expert_freq = np.ones(self.num_experts) / self.num_experts
        
        # 目标：均匀分布
        target = 1.0 / self.num_experts
        loss = np.sum((expert_freq - target) ** 2)
        
        return loss


class MoETransformerBlock:
    """带 MoE 的 Transformer 块"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 num_experts: int, top_k: int = 2):
        from chapter08.transformer_from_scratch import MultiHeadAttention, LayerNorm
        
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray, mask=None) -> Tuple[np.ndarray, float]:
        """前向传播"""
        # 注意力
        attn_out = self.attention.forward(x, mask)
        x = self.norm1.forward(x + attn_out)
        
        # MoE
        moe_out, lb_loss = self.moe.forward(x)
        x = self.norm2.forward(x + moe_out)
        
        return x, lb_loss


def test_moe():
    """测试 MoE"""
    print("=" * 60)
    print("MoE 模型测试")
    print("=" * 60)
    
    d_model = 256
    d_ff = 512
    num_experts = 8
    top_k = 2
    batch_size = 4
    seq_len = 16
    
    # 创建 MoE 层
    moe = MoELayer(d_model, d_ff, num_experts, top_k)
    
    # 测试
    x = np.random.randn(batch_size, seq_len, d_model)
    output, loss = moe.forward(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"负载均衡损失: {loss:.6f}")
    print(f"配置: num_experts={num_experts}, top_k={top_k}")
    
    print("\n✅ MoE 测试通过!")


if __name__ == "__main__":
    test_moe()
