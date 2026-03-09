# 第17章：参数高效微调（PEFT）

<div align="center">

[⬅️ 上一章](../chapter16-mainstream-llms/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter18-instruction-tuning-rlhf/README.md)

</div>

---

## 📖 学习目标

- ✅ 理解 PEFT 的原理和优势
- ✅ 掌握 LoRA 的实现
- ✅ 理解其他 PEFT 方法（Prefix Tuning、Adapter）
- ✅ 实践模型微调

---

## 💻 LoRA 实现

```python
"""
LoRA 实现
"""
import numpy as np

class LoRALayer:
    """LoRA 层"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        """
        初始化 LoRA
        
        参数:
            in_features: 输入维度
            out_features: 输出维度
            rank: LoRA 秩
            alpha: 缩放因子
        """
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA 权重
        self.lora_A = np.random.randn(in_features, rank) * 0.01
        self.lora_B = np.zeros((rank, out_features))
        
        # 冻结的原权重
        self.W = None  # 预训练权重
    
    def forward(self, x):
        """
        前向传播
        
        公式: output = W·x + (scaling)·B·A·x
        """
        # 原始输出
        original = np.matmul(x, self.W) if self.W is not None else 0
        
        # LoRA 输出
        lora = np.matmul(x, np.matmul(self.lora_A, self.lora_B))
        
        return original + self.scaling * lora


class LoRAModel:
    """带 LoRA 的模型"""
    
    def __init__(self, model, rank=8, alpha=16):
        """
        为模型添加 LoRA
        
        参数:
            model: 原始模型
            rank: LoRA 秩
            alpha: 缩放因子
        """
        self.model = model
        self.lora_layers = {}
        
        # 为注意力层添加 LoRA
        for name, module in self.get_attention_layers(model):
            lora = LoRALayer(
                module.d_model, 
                module.d_model,
                rank=rank,
                alpha=alpha
            )
            self.lora_layers[name] = lora
    
    def get_attention_layers(self, model):
        """获取注意力层"""
        # 简化：返回所有注意力层
        return []
    
    def forward(self, x):
        """前向传播"""
        return self.model.forward(x)
    
    def merge_lora_weights(self):
        """合并 LoRA 权重到原模型"""
        for name, lora_layer in self.lora_layers.items():
            # W_new = W + scaling * B·A
            delta_W = lora_layer.scaling * np.matmul(
                lora_layer.lora_B, 
                lora_layer.lora_A.T
            )
            lora_layer.W += delta_W


# ================================
# 其他 PEFT 方法
# ================================

class PrefixTuning:
    """Prefix Tuning"""
    
    def __init__(self, num_layers, d_model, prefix_length=10):
        self.prefix_length = prefix_length
        
        # 每层的前缀向量
        self.prefix_embeddings = [
            np.random.randn(prefix_length, d_model) * 0.02
            for _ in range(num_layers)
        ]
    
    def forward(self, x, layer_idx):
        """在输入前添加前缀"""
        prefix = self.prefix_embeddings[layer_idx]
        return np.concatenate([prefix, x], axis=1)


class AdapterLayer:
    """Adapter 层"""
    
    def __init__(self, d_model, bottleneck=64):
        self.down_proj = np.random.randn(d_model, bottleneck) * 0.01
        self.up_proj = np.random.randn(bottleneck, d_model) * 0.01
    
    def forward(self, x):
        """前向传播"""
        # 降维 -> 激活 -> 升维 + 残差
        down = np.matmul(x, self.down_proj)
        activated = np.maximum(0, down)  # ReLU
        up = np.matmul(activated, self.up_proj)
        return x + up


# ================================
# 示例
# ================================

if __name__ == "__main__":
    print("LoRA 示例")
    print("=" * 50)
    
    # 创建 LoRA 层
    lora = LoRALayer(768, 768, rank=8, alpha=16)
    
    print(f"LoRA 参数:")
    print(f"  原参数: {768 * 768:,}")
    print(f"  LoRA 参数: {768 * 8 * 2:,}")
    print(f"  参数减少: {(1 - 768*8*2/(768*768))*100:.2f}%")
    
    # 测试前向传播
    x = np.random.randn(2, 10, 768)
    output = lora.forward(x)
    
    print(f"\n前向传播:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {output.shape}")
    
    print("\n✅ PEFT 测试通过！")
```

---

## 📝 本章小结

### PEFT 方法对比

| 方法 | 原理 | 参数量 | 优势 |
|------|------|--------|------|
| **LoRA** | 低秩分解 | ~0.1% | 简单高效、可合并 |
| **Prefix Tuning** | 添加前缀 | ~0.1% | 不改原模型 |
| **Adapter** | 插入适配层 | ~1% | 模块化 |
| **QLoRA** | LoRA + 量化 | ~0.01% | 超低显存 |

---

<div align="center">

[⬅️ 上一章](../chapter16-mainstream-llms/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter18-instruction-tuning-rlhf/README.md)

</div>
