"""
第17章：参数高效微调（PEFT）实现
LoRA、Prefix Tuning、Adapter 等方法
"""
import numpy as np
from typing import Optional, List


class LoRALayer:
    """LoRA 层"""
    
    def __init__(self, in_features: int, out_features: int, 
                 rank: int = 8, alpha: float = 16.0):
        """
        参数:
            in_features: 输入维度
            out_features: 输出维度
            rank: LoRA 秩
            alpha: 缩放因子
        """
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA 权重：A 随机初始化，B 初始化为零
        self.lora_A = np.random.randn(in_features, rank) * 0.01
        self.lora_B = np.zeros((rank, out_features))
        
        # 原始权重（冻结）
        self.weight = None
    
    def set_weight(self, weight: np.ndarray):
        """设置原始权重"""
        self.weight = weight
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        输出 = W·x + (alpha/r)·B·A·x
        """
        # 原始输出
        original = np.matmul(x, self.weight) if self.weight is not None else 0
        
        # LoRA 增量
        lora_out = np.matmul(x, np.matmul(self.lora_A, self.lora_B))
        
        return original + self.scaling * lora_out
    
    def get_delta_weight(self) -> np.ndarray:
        """获取 LoRA 增量权重"""
        return self.scaling * np.matmul(self.lora_B, self.lora_A.T)
    
    def merge_weights(self):
        """合并 LoRA 权重到原始权重"""
        if self.weight is not None:
            self.weight = self.weight + self.get_delta_weight()


class LoRALinear:
    """带 LoRA 的线性层"""
    
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 16.0):
        self.in_features = in_features
        self.out_features = out_features
        
        # 原始权重
        self.weight = np.random.randn(in_features, out_features) * 0.02
        self.bias = np.zeros(out_features)
        
        # LoRA
        self.lora = LoRALayer(in_features, out_features, rank, alpha)
        self.lora.set_weight(self.weight)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        return self.lora.forward(x) + self.bias


class LoRAConfig:
    """LoRA 配置"""
    
    def __init__(self, r: int = 8, lora_alpha: float = 16.0,
                 target_modules: List[str] = None, lora_dropout: float = 0.0):
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.lora_dropout = lora_dropout


class PrefixTuning:
    """Prefix Tuning"""
    
    def __init__(self, num_layers: int, d_model: int, prefix_length: int = 10):
        self.prefix_length = prefix_length
        
        # 每层的前缀向量
        self.prefix_keys = [
            np.random.randn(prefix_length, d_model) * 0.02
            for _ in range(num_layers)
        ]
        self.prefix_values = [
            np.random.randn(prefix_length, d_model) * 0.02
            for _ in range(num_layers)
        ]
    
    def get_prefix(self, layer_idx: int) -> tuple:
        """获取指定层的前缀"""
        return self.prefix_keys[layer_idx], self.prefix_values[layer_idx]
    
    def forward_with_prefix(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """在输入前添加前缀"""
        prefix_k, prefix_v = self.get_prefix(layer_idx)
        # 简化：直接拼接
        return np.concatenate([prefix_k[np.newaxis, :, :], x], axis=1)


class AdapterLayer:
    """Adapter 层"""
    
    def __init__(self, d_model: int, bottleneck: int = 64):
        self.down_proj = np.random.randn(d_model, bottleneck) * 0.01
        self.up_proj = np.random.randn(bottleneck, d_model) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播: x + Up(ReLU(Down(x)))"""
        down = np.matmul(x, self.down_proj)
        activated = np.maximum(0, down)  # ReLU
        up = np.matmul(activated, self.up_proj)
        return x + up


class AdapterTransformerBlock:
    """带 Adapter 的 Transformer 块"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 adapter_bottleneck: int = 64):
        from chapter08.transformer_from_scratch import MultiHeadAttention, LayerNorm
        
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Adapter 层
        self.adapter1 = AdapterLayer(d_model, adapter_bottleneck)
        self.adapter2 = AdapterLayer(d_model, adapter_bottleneck)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Attention + Adapter
        residual = x
        x = self.norm1.forward(x)
        x = self.attention.forward(x, mask)
        x = self.adapter1.forward(x)
        x = residual + x
        
        # FFN + Adapter
        residual = x
        x = self.norm2.forward(x)
        x = self.ffn.forward(x)
        x = self.adapter2.forward(x)
        x = residual + x
        
        return x


class FeedForward:
    """简单前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(np.maximum(0, np.matmul(x, self.W1)), self.W2)


def compare_parameters():
    """对比不同方法的参数量"""
    d_model = 4096
    
    print("参数量对比 (d_model=4096):")
    print("-" * 50)
    
    # 全参数
    full_params = d_model * d_model
    print(f"全参数线性层: {full_params:,}")
    
    # LoRA (不同 rank)
    for rank in [4, 8, 16, 32]:
        lora_params = 2 * d_model * rank
        ratio = lora_params / full_params * 100
        print(f"LoRA (r={rank}): {lora_params:,} ({ratio:.2f}%)")
    
    # Adapter
    bottleneck = 64
    adapter_params = 2 * d_model * bottleneck
    ratio = adapter_params / full_params * 100
    print(f"Adapter (bottleneck={bottleneck}): {adapter_params:,} ({ratio:.2f}%)")
    
    # Prefix Tuning
    prefix_len = 10
    prefix_params = prefix_len * d_model
    ratio = prefix_params / full_params * 100
    print(f"Prefix Tuning (len={prefix_len}): {prefix_params:,} ({ratio:.2f}%)")


def test_lora():
    """测试 LoRA"""
    print("=" * 60)
    print("LoRA 测试")
    print("=" * 60)
    
    # 创建 LoRA 线性层
    lora_linear = LoRALinear(768, 768, rank=8, alpha=16.0)
    
    # 测试
    x = np.random.randn(2, 10, 768)
    output = lora_linear.forward(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 参数量对比
    original_params = 768 * 768
    lora_params = 2 * 768 * 8
    
    print(f"\n原始参数: {original_params:,}")
    print(f"LoRA 参数: {lora_params:,}")
    print(f"参数减少: {(1 - lora_params/original_params)*100:.2f}%")
    
    print("\n✅ LoRA 测试通过!")


if __name__ == "__main__":
    test_lora()
    print()
    compare_parameters()
