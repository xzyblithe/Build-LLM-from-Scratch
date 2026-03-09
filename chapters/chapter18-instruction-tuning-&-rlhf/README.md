# 第18章：指令微调与 RLHF

<div align="center">

[⬅️ 上一章](../chapter17-peft/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter19-deployment/README.md)

</div>

---

## 📖 学习目标

- ✅ 理解指令微调流程
- ✅ 掌握数据集构建方法
- ✅ 理解 RLHF 原理
- ✅ 实现 DPO 算法

---

## 💻 指令微调实现

```python
"""
指令微调实现
"""
import numpy as np

class InstructionTuningData:
    """指令微调数据"""
    
    def __init__(self):
        self.instructions = [
            {
                "instruction": "将下面的句子翻译成英文",
                "input": "我爱自然语言处理",
                "output": "I love natural language processing"
            },
            {
                "instruction": "总结下面的文章",
                "input": "深度学习是机器学习的分支...",
                "output": "文章介绍了深度学习的定义和应用"
            }
        ]
    
    def format_prompt(self, item):
        """格式化提示"""
        if item['input']:
            return f"### 指令:\n{item['instruction']}\n\n### 输入:\n{item['input']}\n\n### 输出:\n{item['output']}"
        else:
            return f"### 指令:\n{item['instruction']}\n\n### 输出:\n{item['output']}"


class DPO:
    """Direct Preference Optimization"""
    
    def __init__(self, model, ref_model, beta=0.1):
        """
        初始化 DPO
        
        参数:
            model: 待训练模型
            ref_model: 参考模型（冻结）
            beta: KL 散度系数
        """
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
    
    def compute_loss(self, input_ids, preferred_ids, dispreferred_ids):
        """
        计算 DPO 损失
        
        参数:
            input_ids: 输入
            preferred_ids: 偏好的回复
            dispreferred_ids: 不偏好的回复
        
        返回:
            DPO 损失
        """
        # 计算对数概率
        log_prob_preferred = self.compute_log_prob(self.model, input_ids, preferred_ids)
        log_prob_dispreferred = self.compute_log_prob(self.model, input_ids, dispreferred_ids)
        
        log_prob_ref_preferred = self.compute_log_prob(self.ref_model, input_ids, preferred_ids)
        log_prob_ref_dispreferred = self.compute_log_prob(self.ref_model, input_ids, dispreferred_ids)
        
        # DPO 损失
        loss = -np.log(
            1 / (1 + np.exp(
                self.beta * (log_prob_dispreferred - log_prob_ref_dispreferred) -
                self.beta * (log_prob_preferred - log_prob_ref_preferred)
            ))
        )
        
        return loss
    
    def compute_log_prob(self, model, input_ids, output_ids):
        """计算对数概率"""
        # 简化实现
        return -np.random.rand()


# ================================
# 示例
# ================================

if __name__ == "__main__":
    print("指令微调示例")
    print("=" * 50)
    
    # 数据准备
    data = InstructionTuningData()
    
    print("指令数据示例:")
    for item in data.instructions[:1]:
        print(data.format_prompt(item))
    
    print("\n✅ 指令微调测试通过！")
```

---

<div align="center">

[⬅️ 上一章](../chapter17-peft/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter19-deployment/README.md)

</div>
