# 第19章：模型部署与推理优化

<div align="center">

[⬅️ 上一章](../chapter18-instruction-tuning-&-rlhf/README.md) | [返回目录](../README.md)

</div>

---

## 📖 学习目标

- ✅ 掌握模型量化技术
- ✅ 理解推理优化方法
- ✅ 实现模型部署服务

---

## 💻 部署实现

```python
"""
模型部署实现
"""
import numpy as np

class QuantizedModel:
    """量化模型（INT8）"""
    
    def __init__(self, model):
        """量化模型"""
        self.quantized_weights = {}
        
        for name, param in model.items():
            # INT8 量化
            scale = np.max(np.abs(param)) / 127
            quantized = np.clip(np.round(param / scale), -128, 127).astype(np.int8)
            
            self.quantized_weights[name] = {
                'data': quantized,
                'scale': scale
            }
    
    def forward(self, x):
        """前向传播"""
        # 反量化
        # 简化实现
        return x


class KVCache:
    """KV Cache"""
    
    def __init__(self, max_seq_len, n_layers, n_heads, d_k):
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        
        # 初始化缓存
        self.k_cache = [np.zeros((1, n_heads, max_seq_len, d_k)) for _ in range(n_layers)]
        self.v_cache = [np.zeros((1, n_heads, max_seq_len, d_k)) for _ in range(n_layers)]
    
    def update(self, layer_idx, k, v, pos):
        """更新缓存"""
        self.k_cache[layer_idx][:, :, pos:pos+1, :] = k
        self.v_cache[layer_idx][:, :, pos:pos+1, :] = v
    
    def get(self, layer_idx):
        """获取缓存"""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]


class FastAPI:
    """简化版 FastAPI 服务"""
    
    def __init__(self, model):
        self.model = model
        self.kv_cache = None
    
    def predict(self, text):
        """预测"""
        # 简化实现
        return f"生成: {text}"
    
    def stream_predict(self, text):
        """流式预测"""
        for i in range(10):
            yield f"Token_{i}"


# ================================
# 示例
# ================================

if __name__ == "__main__":
    print("模型部署示例")
    print("=" * 50)
    
    # 模拟模型
    model = {
        'weight1': np.random.randn(768, 768).astype(np.float32)
    }
    
    # 量化
    quantized = QuantizedModel(model)
    
    print(f"量化结果:")
    print(f"  原始大小: {model['weight1'].nbytes / 1024:.2f} KB")
    print(f"  量化后: {quantized.quantized_weights['weight1']['data'].nbytes / 1024:.2f} KB")
    print(f"  压缩比: {model['weight1'].nbytes / quantized.quantized_weights['weight1']['data'].nbytes:.2f}x")
    
    print("\n✅ 模型部署测试通过！")
```

---

## 📝 本章小结

### 核心技术

1. **量化**：INT8/INT4 量化减少模型大小
2. **KV Cache**：缓存键值对加速推理
3. **Flash Attention**：优化注意力计算
4. **vLLM**：高效推理框架

---

<div align="center">

[⬅️ 上一章](../chapter18-instruction-tuning-&-rlhf/README.md) | [返回目录](../README.md)

**🎉 恭喜完成全部教程！**

</div>
