"""
第18章：指令微调与 RLHF 实现
SFT、DPO、PPO 等对齐方法
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class InstructionData:
    """指令数据格式"""
    instruction: str
    input: str = ""
    output: str = ""


class InstructionFormatter:
    """指令数据格式化"""
    
    @staticmethod
    def alpaca_format(item: InstructionData) -> str:
        """Alpaca 格式"""
        if item.input:
            return f"""### 指令:
{item.instruction}

### 输入:
{item.input}

### 输出:
{item.output}"""
        else:
            return f"""### 指令:
{item.instruction}

### 输出:
{item.output}"""
    
    @staticmethod
    def chatml_format(item: InstructionData) -> str:
        """ChatML 格式"""
        return f"""<|im_start|>user
{item.instruction}
{item.input}<|im_end|>
<|im_start|>assistant
{item.output}<|im_end|>"""


class PreferenceData:
    """偏好数据"""
    
    def __init__(self, prompt: str, chosen: str, rejected: str):
        self.prompt = prompt
        self.chosen = chosen
        self.rejected = rejected


class DPOTrainer:
    """Direct Preference Optimization 训练器"""
    
    def __init__(self, beta: float = 0.1, learning_rate: float = 1e-6):
        """
        参数:
            beta: KL 散度系数
            learning_rate: 学习率
        """
        self.beta = beta
        self.learning_rate = learning_rate
    
    def compute_dpo_loss(self, 
                         policy_chosen_logp: float,
                         policy_rejected_logp: float,
                         reference_chosen_logp: float,
                         reference_rejected_logp: float) -> float:
        """
        计算 DPO 损失
        
        L = -log(σ(β * (log(π_chosen/π_ref_chosen) - log(π_rejected/π_ref_rejected))))
        """
        # 计算对数比率
        chosen_ratio = policy_chosen_logp - reference_chosen_logp
        rejected_ratio = policy_rejected_logp - reference_rejected_logp
        
        # DPO 目标
        logits = self.beta * (chosen_ratio - rejected_ratio)
        
        # Sigmoid 交叉熵
        loss = -np.log(1 / (1 + np.exp(-logits)))
        
        return loss
    
    def train_step(self, batch: List[PreferenceData]) -> float:
        """单步训练"""
        losses = []
        
        for item in batch:
            # 模拟对数概率（实际中需要模型计算）
            policy_chosen_logp = np.random.randn() - 0.5
            policy_rejected_logp = np.random.randn() - 1.0
            reference_chosen_logp = np.random.randn() - 0.3
            reference_rejected_logp = np.random.randn() - 0.8
            
            loss = self.compute_dpo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                reference_chosen_logp,
                reference_rejected_logp
            )
            losses.append(loss)
        
        return np.mean(losses)


class RewardModel:
    """奖励模型"""
    
    def __init__(self, d_model: int = 768):
        self.d_model = d_model
        # 简化的模型参数
        self.embedding = np.random.randn(50000, d_model) * 0.02
        self.reward_head = np.random.randn(d_model, 1) * 0.02
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """计算奖励分数"""
        # 简化实现
        hidden = self.embedding[input_ids]
        # 取最后一个 token
        last_hidden = hidden[:, -1, :]
        reward = np.matmul(last_hidden, self.reward_head)
        return reward.squeeze(-1)
    
    def compute_preference_loss(self, chosen_rewards: np.ndarray, 
                                rejected_rewards: np.ndarray) -> float:
        """计算偏好损失"""
        diff = chosen_rewards - rejected_rewards
        loss = -np.mean(np.log(1 / (1 + np.exp(-diff))))
        return loss


class PPOTrainer:
    """PPO 训练器"""
    
    def __init__(self, clip_range: float = 0.2, kl_coef: float = 0.1,
                 gamma: float = 0.99):
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.gamma = gamma
    
    def compute_ppo_loss(self, 
                         old_log_probs: np.ndarray,
                         new_log_probs: np.ndarray,
                         advantages: np.ndarray) -> float:
        """
        计算 PPO 损失
        
        L = min(r * A, clip(r, 1-ε, 1+ε) * A)
        其中 r = π_new / π_old
        """
        # 计算比率
        ratio = np.exp(new_log_probs - old_log_probs)
        
        # 裁剪
        clipped_ratio = np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
        
        # 取较小值
        loss = -np.minimum(ratio * advantages, clipped_ratio * advantages)
        
        return np.mean(loss)
    
    def compute_advantages(self, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        """计算优势函数"""
        # 简化的 GAE 计算
        advantages = rewards - values
        return advantages


class SFTTrainer:
    """有监督微调训练器"""
    
    def __init__(self, learning_rate: float = 1e-5, max_length: int = 512):
        self.learning_rate = learning_rate
        self.max_length = max_length
    
    def prepare_inputs(self, item: InstructionData) -> Dict:
        """准备模型输入"""
        # 格式化
        prompt = InstructionFormatter.alpaca_format(item)
        
        # 简化：返回模拟的 token ids
        input_ids = np.random.randint(0, 50000, (min(len(prompt), self.max_length),))
        
        # 标签：只在输出部分计算损失
        # 输入部分设为 -100（忽略）
        output_start = prompt.find("### 输出:")
        labels = input_ids.copy()
        # 简化处理
        labels[:output_start // 10] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': np.ones_like(input_ids)
        }


def demonstrate_alignment_pipeline():
    """演示对齐流程"""
    print("=" * 60)
    print("LLM 对齐流程演示")
    print("=" * 60)
    
    # ========== 阶段 1: SFT ==========
    print("\n【阶段 1】有监督微调 (SFT)")
    sft_trainer = SFTTrainer()
    
    instruction_data = InstructionData(
        instruction="将下面的句子翻译成英文",
        input="我喜欢学习人工智能",
        output="I like learning artificial intelligence."
    )
    
    inputs = sft_trainer.prepare_inputs(instruction_data)
    print(f"输入 IDs 形状: {inputs['input_ids'].shape}")
    print(f"标签形状: {inputs['labels'].shape}")
    
    # ========== 阶段 2: 奖励模型 ==========
    print("\n【阶段 2】奖励模型训练")
    reward_model = RewardModel(d_model=256)
    
    # 模拟偏好数据
    chosen_ids = np.random.randint(0, 50000, (4, 32))
    rejected_ids = np.random.randint(0, 50000, (4, 32))
    
    chosen_rewards = reward_model.forward(chosen_ids)
    rejected_rewards = reward_model.forward(rejected_ids)
    
    pref_loss = reward_model.compute_preference_loss(chosen_rewards, rejected_rewards)
    print(f"偏好损失: {pref_loss:.4f}")
    
    # ========== 阶段 3: DPO ==========
    print("\n【阶段 3】DPO 对齐")
    dpo_trainer = DPOTrainer(beta=0.1)
    
    preference_data = [
        PreferenceData("什么是AI？", "AI是人工智能...", "AI是一个技术"),
        PreferenceData("如何学习编程？", "建议从Python开始...", "看书就行")
    ]
    
    dpo_loss = dpo_trainer.train_step(preference_data)
    print(f"DPO 损失: {dpo_loss:.4f}")
    
    print("\n✅ 对齐流程演示完成!")


def compare_alignment_methods():
    """对比对齐方法"""
    print("\n" + "=" * 60)
    print("对齐方法对比")
    print("=" * 60)
    
    methods = [
        ("SFT", "有监督微调", "基础对齐", "低"),
        ("RLHF (PPO)", "强化学习", "高级对齐", "高"),
        ("DPO", "直接偏好优化", "高级对齐", "中"),
        ("DPO + LoRA", "参数高效 DPO", "高级对齐", "低"),
    ]
    
    print(f"\n{'方法':<15} {'原理':<15} {'对齐程度':<12} {'计算成本'}")
    print("-" * 60)
    for method, principle, level, cost in methods:
        print(f"{method:<15} {principle:<15} {level:<12} {cost}")


if __name__ == "__main__":
    demonstrate_alignment_pipeline()
    compare_alignment_methods()
