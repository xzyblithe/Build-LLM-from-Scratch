# 第18章：指令微调与 RLHF

<div align="center">

[⬅️ 上一章](../chapter17-peft/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter19-deployment/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解指令微调的流程与数据构建
- ✅ 掌握 SFT（有监督微调）方法
- ✅ 理解 RLHF 的三阶段训练流程
- ✅ 实现 DPO 算法
- ✅ 了解 PPO 算法原理

---

## 🎯 对齐技术概览

### 为什么需要对齐？

```
预训练模型的问题：

1. 续写而非回答
   输入: "法国的首都是哪里？"
   输出: "英国的首都是哪里？德国的首都是哪里？..."
   （模型在续写，而不是回答）

2. 不遵循指令
   输入: "请用三句话介绍自己"
   输出: 可能输出任何内容

3. 安全性问题
   可能生成有害、偏见、虚假内容

解决方案：对齐训练
- 指令微调（SFT）：让模型学会遵循指令
- RLHF：让模型输出符合人类偏好
```

### 对齐技术发展

```
时间线：

2022.01 - InstructGPT（OpenAI）
          提出 RLHF 三阶段方法

2023.02 - LLaMA（Meta）
          开源基座模型

2023.04 - Dolly（Databricks）
          开源指令数据

2023.05 - DPO 论文
          简化 RLHF，无需奖励模型

2023.07 - LLaMA-2-Chat
          开源 RLHF 全流程

2024.01 - Direct Preference Optimization
          成为对齐主流方法
```

---

## 1. 指令微调（SFT）

### 1.1 数据集构建

```python
"""
指令微调数据集构建
"""
import json
import numpy as np

class InstructionDataset:
    """指令微调数据集"""
    
    def __init__(self):
        self.data = []
    
    def add_item(self, instruction, input_text="", output_text=""):
        """添加数据项"""
        self.data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    
    def format_prompt(self, item, template="alpaca"):
        """
        格式化为模型输入
        
        支持多种模板:
        - alpaca: Alpaca 格式
        - chatml: ChatML 格式
        - vicuna: Vicuna 格式
        """
        if template == "alpaca":
            if item['input']:
                return f"""### 指令:
{item['instruction']}

### 输入:
{item['input']}

### 输出:
{item['output']}"""
            else:
                return f"""### 指令:
{item['instruction']}

### 输出:
{item['output']}"""
        
        elif template == "chatml":
            return f"""<|im_start|>user
{item['instruction']}
{item['input']}<|im_end|>
<|im_start|>assistant
{item['output']}<|im_end|>"""
        
        elif template == "vicuna":
            return f"""USER: {item['instruction']} {item['input']}
ASSISTANT: {item['output']}</s>"""
        
        return ""
    
    def save_json(self, filepath):
        """保存为 JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in self.data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def load_json(self, filepath):
        """加载 JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))


# 示例：构建数据集
dataset = InstructionDataset()

# 任务类型示例
tasks = [
    # 翻译任务
    {
        "instruction": "将下面的句子翻译成英文",
        "input": "我喜欢学习人工智能",
        "output": "I like learning artificial intelligence."
    },
    # 问答任务
    {
        "instruction": "回答下面的问题",
        "input": "什么是深度学习？",
        "output": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的层次化表示。"
    },
    # 摘要任务
    {
        "instruction": "总结下面的文章",
        "input": "自然语言处理（NLP）是人工智能的一个重要分支...",
        "output": "文章介绍了自然语言处理的定义、发展历程和应用领域。"
    },
    # 分类任务
    {
        "instruction": "判断下面句子的情感极性",
        "input": "这部电影太精彩了！",
        "output": "正面情感"
    },
    # 代码任务
    {
        "instruction": "编写一个Python函数，计算斐波那契数列",
        "input": "",
        "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    }
]

for task in tasks:
    dataset.add_item(**task)

print("数据集示例:")
print(dataset.format_prompt(dataset.data[0]))
```

### 1.2 开源指令数据集

```python
"""
常用开源指令数据集
"""

INSTRUCTION_DATASETS = {
    # 通用指令数据集
    "alpaca": {
        "name": "Stanford Alpaca",
        "size": "52K",
        "language": "英文",
        "source": "tatsu-lab/alpaca",
        "description": "使用 text-davinci-003 生成的指令数据"
    },
    
    "alpaca_zh": {
        "name": "Alpaca 中文版",
        "size": "52K",
        "language": "中文",
        "source": "shareAI/Alpaca",
        "description": "Alpaca 的中文翻译版本"
    },
    
    "dolly": {
        "name": "Databricks Dolly",
        "size": "15K",
        "language": "英文",
        "source": "databricks/databricks-dolly-15k",
        "description": "人工标注的高质量指令数据"
    },
    
    "belle": {
        "name": "BELLE",
        "size": "200万+",
        "language": "中文",
        "source": "BelleGroup/train_2M_CN",
        "description": "大规模中文指令数据集"
    },
    
    "firefly": {
        "name": "Firefly",
        "size": "160万",
        "language": "中文",
        "source": "YeungNLP/firefly-train-1.1M",
        "description": "中文多任务指令数据集"
    },
    
    # 对话数据集
    "sharegpt": {
        "name": "ShareGPT",
        "size": "90K+",
        "language": "多语言",
        "source": "RyokoAI/ShareGPT52K",
        "description": "真实用户对话数据"
    },
    
    # 代码数据集
    "code_alpaca": {
        "name": "Code Alpaca",
        "size": "20K",
        "language": "代码",
        "source": "sahil2801/CodeAlpaca-20k",
        "description": "代码生成指令数据"
    }
}

# 打印数据集信息
print("开源指令数据集:\n")
for key, info in INSTRUCTION_DATASETS.items():
    print(f"{info['name']}: {info['size']} ({info['language']})")
    print(f"  来源: {info['source']}")
    print(f"  说明: {info['description']}\n")
```

### 1.3 SFT 训练流程

```python
import numpy as np

class SFTTrainer:
    """有监督微调训练器"""
    
    def __init__(self, model, tokenizer, learning_rate=1e-5, max_length=512):
        """
        参数:
            model: 语言模型
            tokenizer: 分词器
            learning_rate: 学习率
            max_length: 最大序列长度
        """
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.max_length = max_length
    
    def prepare_batch(self, batch_data):
        """
        准备训练批次
        
        指令微调的关键:
        - 只在输出部分计算损失
        - 输入（指令+问题）部分的标签设为 -100（忽略）
        """
        input_ids = []
        attention_mask = []
        labels = []
        
        for item in batch_data:
            # 构建完整输入
            prompt = self.format_prompt(item)
            full_text = prompt + item['output']
            
            # Tokenize
            tokenized = self.tokenizer(full_text, max_length=self.max_length)
            input_ids.append(tokenized['input_ids'])
            attention_mask.append(tokenized['attention_mask'])
            
            # 标签：输入部分用 -100 掩盖
            prompt_tokens = self.tokenizer(prompt)['input_ids']
            prompt_len = len(prompt_tokens)
            
            label = tokenized['input_ids'].copy()
            label[:prompt_len] = [-100] * prompt_len  # 忽略输入部分
            labels.append(label)
        
        return {
            'input_ids': np.array(input_ids),
            'attention_mask': np.array(attention_mask),
            'labels': np.array(labels)
        }
    
    def format_prompt(self, item):
        """格式化提示"""
        if item['input']:
            return f"### 指令:\n{item['instruction']}\n\n### 输入:\n{item['input']}\n\n### 输出:\n"
        else:
            return f"### 指令:\n{item['instruction']}\n\n### 输出:\n"
    
    def compute_loss(self, logits, labels):
        """
        计算损失
        
        只在非 -100 的位置计算交叉熵损失
        """
        # 找到有效位置
        valid_mask = labels != -100
        
        # 展平
        logits_flat = logits[valid_mask]
        labels_flat = labels[valid_mask]
        
        # 交叉熵损失
        log_probs = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
        
        loss = -log_probs[np.arange(len(labels_flat)), labels_flat].mean()
        
        return loss
    
    def train_step(self, batch):
        """单步训练"""
        # 前向传播
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # 获取 logits（简化）
        logits = self.model.forward(input_ids)
        
        # 计算损失
        loss = self.compute_loss(logits, labels)
        
        # 反向传播（简化）
        # gradients = self.backward(loss)
        # self.model.update(self.learning_rate, gradients)
        
        return loss
    
    def train(self, dataset, epochs=3, batch_size=8):
        """完整训练"""
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # 打乱数据
            indices = np.random.permutation(len(dataset.data))
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_data = [dataset.data[j] for j in batch_indices]
                
                batch = self.prepare_batch(batch_data)
                loss = self.train_step(batch)
                
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


# 简化的模型类
class SimpleLLM:
    """简化的语言模型"""
    
    def __init__(self, vocab_size=10000, d_model=512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.02
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        hidden = self.embedding[input_ids]
        logits = np.matmul(hidden, self.output_proj)
        return logits


# 示例训练
print("SFT 训练示例:")
model = SimpleLLM(vocab_size=1000, d_model=256)
tokenizer = lambda x, **kw: {'input_ids': np.random.randint(0, 1000, (100,)), 
                             'attention_mask': np.ones(100)}
trainer = SFTTrainer(model, tokenizer)
print("SFT Trainer 初始化完成")
```

---

## 2. RLHF 原理

### 2.1 RLHF 三阶段

```
RLHF (Reinforcement Learning from Human Feedback) 三阶段:

阶段1: 有监督微调（SFT）
├── 使用指令数据微调预训练模型
├── 让模型学会遵循指令
└── 得到 SFT 模型

阶段2: 奖励模型训练（RM）
├── 收集人类偏好数据
│   输入: prompt
│   输出: 两个回复 + 人类选择
├── 训练奖励模型打分
└── 得到 Reward Model

阶段3: 强化学习（PPO）
├── 使用 PPO 算法优化
├── 目标: 最大化奖励
├── 约束: KL 散度（不偏离太远）
└── 得到最终对齐模型
```

### 2.2 奖励模型

```python
import numpy as np

class RewardModel:
    """奖励模型"""
    
    def __init__(self, d_model=768, vocab_size=50000):
        """
        奖励模型: 输入文本 -> 输出标量分数
        
        通常使用与 SFT 相同的架构，
        但最后用一个线性层输出标量
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 共享的 Transformer 编码器（简化）
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # 奖励头
        self.reward_head = np.random.randn(d_model, 1) * 0.02
    
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播
        
        参数:
            input_ids: 输入 token IDs
        
        返回:
            reward: 标量奖励值
        """
        # Embedding
        hidden = self.embedding[input_ids]
        
        # 取最后一个 token 的表示
        last_hidden = hidden[:, -1, :]
        
        # 输出奖励
        reward = np.matmul(last_hidden, self.reward_head)
        
        return reward.squeeze(-1)
    
    def compute_preference_loss(self, chosen_rewards, rejected_rewards):
        """
        计算偏好损失
        
        目标: 让 chosen 的奖励高于 rejected
        
        Loss = -log(sigmoid(reward_chosen - reward_rejected))
        """
        diff = chosen_rewards - rejected_rewards
        loss = -np.log(1 / (1 + np.exp(-diff)))
        return np.mean(loss)


class PreferenceDataset:
    """偏好数据集"""
    
    def __init__(self):
        self.data = []
    
    def add_item(self, prompt, chosen_response, rejected_response):
        """添加偏好数据"""
        self.data.append({
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response
        })
    
    def get_batch(self, batch_size=8):
        """获取批次"""
        indices = np.random.choice(len(self.data), batch_size, replace=True)
        return [self.data[i] for i in indices]


# 训练奖励模型
def train_reward_model():
    """训练奖励模型"""
    
    # 初始化
    reward_model = RewardModel(d_model=256, vocab_size=1000)
    dataset = PreferenceDataset()
    
    # 添加示例数据
    preferences = [
        {
            "prompt": "解释什么是机器学习",
            "chosen": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出决策或预测。",
            "rejected": "机器学习是一种技术。"
        },
        {
            "prompt": "如何学习编程？",
            "chosen": "建议从 Python 开始，通过实践项目学习，逐步掌握基础概念和高级技巧。",
            "rejected": "多看书就行了。"
        }
    ]
    
    for p in preferences:
        dataset.add_item(**p)
    
    # 训练
    learning_rate = 1e-5
    epochs = 10
    
    for epoch in range(epochs):
        batch = dataset.get_batch(batch_size=2)
        
        # 简化：随机生成模拟奖励
        chosen_rewards = np.random.randn(2)
        rejected_rewards = np.random.randn(2) - 0.5  # 让 chosen 通常更高
        
        loss = reward_model.compute_preference_loss(chosen_rewards, rejected_rewards)
        
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    print("\n奖励模型训练完成")
    return reward_model

reward_model = train_reward_model()
```

### 2.3 PPO 算法

```python
import numpy as np

class PPOTrainer:
    """PPO 训练器"""
    
    def __init__(self, policy_model, ref_model, reward_model,
                 kl_coef=0.1, clip_range=0.2, gamma=0.99):
        """
        参数:
            policy_model: 策略模型（待优化）
            ref_model: 参考模型（冻结，用于 KL 约束）
            reward_model: 奖励模型
            kl_coef: KL 散度系数
            clip_range: PPO 裁剪范围
            gamma: 折扣因子
        """
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.gamma = gamma
    
    def compute_kl_divergence(self, policy_logits, ref_logits):
        """
        计算 KL 散度
        
        KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
        """
        # Softmax
        policy_probs = np.exp(policy_logits - np.max(policy_logits, axis=-1, keepdims=True))
        policy_probs = policy_probs / np.sum(policy_probs, axis=-1, keepdims=True)
        
        ref_probs = np.exp(ref_logits - np.max(ref_logits, axis=-1, keepdims=True))
        ref_probs = ref_probs / np.sum(ref_probs, axis=-1, keepdims=True)
        
        # KL 散度
        kl = np.sum(policy_probs * (np.log(policy_probs + 1e-10) - np.log(ref_probs + 1e-10)), axis=-1)
        
        return np.mean(kl)
    
    def compute_ppo_loss(self, old_log_probs, new_log_probs, advantages):
        """
        计算 PPO 损失
        
        PPO-Clip 目标:
        L = min(r * A, clip(r, 1-ε, 1+ε) * A)
        
        其中 r = π_new / π_old
        """
        # 计算比率
        ratio = np.exp(new_log_probs - old_log_probs)
        
        # 裁剪
        clipped_ratio = np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
        
        # 取较小值
        loss = -np.min(ratio * advantages, clipped_ratio * advantages)
        
        return np.mean(loss)
    
    def generate_response(self, prompt, max_length=100):
        """生成回复"""
        # 简化实现
        return prompt + " [生成的回复]"
    
    def train_step(self, prompts):
        """
        单步训练
        
        流程:
        1. 用 policy_model 生成回复
        2. 计算奖励
        3. 计算 KL 惩罚
        4. 更新策略
        """
        total_loss = 0
        
        for prompt in prompts:
            # 生成回复
            response = self.generate_response(prompt)
            
            # 计算奖励（简化）
            reward = np.random.rand()  # 实际用 reward_model
            
            # 计算 KL 散度（简化）
            kl = np.random.rand() * 0.1  # 实际计算
            
            # 总目标 = 奖励 - KL 惩罚
            objective = reward - self.kl_coef * kl
            
            total_loss += -objective  # 最小化负目标
        
        return total_loss / len(prompts)
    
    def train(self, prompts, epochs=10):
        """完整训练"""
        for epoch in range(epochs):
            loss = self.train_step(prompts)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")


# 简化的模型
class SimplePolicyModel:
    def forward(self, x):
        return np.random.randn(len(x), 100)  # 简化的 logits


# PPO 训练示例
print("\nPPO 训练示例:")
policy_model = SimplePolicyModel()
ref_model = SimplePolicyModel()
reward_model = RewardModel(d_model=256, vocab_size=1000)

ppo_trainer = PPOTrainer(policy_model, ref_model, reward_model)

prompts = ["什么是人工智能？", "如何学习编程？"]
ppo_trainer.train(prompts, epochs=5)
```

---

## 3. DPO 算法

### 3.1 DPO 原理

```
DPO (Direct Preference Optimization) 核心思想:

RLHF 问题:
- 需要训练奖励模型
- 需要复杂的 PPO 训练
- 训练不稳定

DPO 解决方案:
- 直接从偏好数据优化策略
- 不需要显式的奖励模型
- 简化为分类问题

数学推导:
从 Bradley-Terry 模型出发:
P(chosen > rejected) = σ(r(x, y_chosen) - r(x, y_rejected))

最优策略 π* 和奖励函数 r 的关系:
r(x, y) = β * log(π*(y|x) / π_ref(y|x))

代入得到 DPO 损失:
L_DPO = -E[log σ(β * (log π(y_chosen|x) - log π_ref(y_chosen|x) 
                         - log π(y_rejected|x) + log π_ref(y_rejected|x)))]
```

### 3.2 DPO 实现

```python
import numpy as np

class DPOTrainer:
    """DPO 训练器"""
    
    def __init__(self, policy_model, ref_model, beta=0.1, learning_rate=1e-6):
        """
        参数:
            policy_model: 待训练策略模型
            ref_model: 参考模型（冻结）
            beta: KL 散度系数
            learning_rate: 学习率
        """
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.beta = beta
        self.learning_rate = learning_rate
    
    def compute_log_prob(self, model, input_ids, labels):
        """
        计算对数概率
        
        参数:
            model: 语言模型
            input_ids: 完整序列（prompt + response）
            labels: 只在 response 部分计算
        
        返回:
            序列的 log probability
        """
        # 前向传播
        logits = model.forward(input_ids)
        
        # 计算 log softmax
        log_probs = logits - np.max(logits, axis=-1, keepdims=True)
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
        
        # 只在有效位置计算
        # 简化：假设 labels 和 input_ids 相同长度
        seq_log_prob = 0
        for t in range(len(input_ids)):
            if labels[t] != -100:  # 有效位置
                seq_log_prob += log_probs[t, labels[t]]
        
        return seq_log_prob
    
    def compute_dpo_loss(self, chosen_log_probs, rejected_log_probs,
                         ref_chosen_log_probs, ref_rejected_log_probs):
        """
        计算 DPO 损失
        
        L = -E[log σ(β * (log(π/π_ref)_chosen - log(π/π_ref)_rejected))]
        """
        # 计算对数比率
        chosen_ratio = chosen_log_probs - ref_chosen_log_probs
        rejected_ratio = rejected_log_probs - ref_rejected_log_probs
        
        # DPO 目标
        logits = self.beta * (chosen_ratio - rejected_ratio)
        
        # Sigmoid 交叉熵
        loss = -np.log(1 / (1 + np.exp(-logits)))
        
        return loss
    
    def train_step(self, batch):
        """
        单步训练
        
        batch 包含:
        - prompt: 输入提示
        - chosen: 偏好的回复
        - rejected: 不偏好的回复
        """
        losses = []
        
        for item in batch:
            prompt = item['prompt']
            chosen = item['chosen']
            rejected = item['rejected']
            
            # 构建完整序列
            chosen_text = prompt + chosen
            rejected_text = prompt + rejected
            
            # 简化：使用随机模拟
            chosen_log_prob = np.random.randn() - 1  # policy log prob
            rejected_log_prob = np.random.randn() - 2
            ref_chosen_log_prob = np.random.randn() - 0.5  # reference log prob
            ref_rejected_log_prob = np.random.randn() - 1.5
            
            # 计算 DPO 损失
            loss = self.compute_dpo_loss(
                chosen_log_prob, rejected_log_prob,
                ref_chosen_log_prob, ref_rejected_log_prob
            )
            losses.append(loss)
        
        return np.mean(losses)
    
    def train(self, dataset, epochs=3, batch_size=8):
        """完整训练"""
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            indices = np.random.permutation(len(dataset.data))
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch = [dataset.data[j] for j in batch_indices]
                
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


# DPO 完整示例
print("\n" + "="*50)
print("DPO 训练示例")
print("="*50)

# 创建模型
policy_model = SimplePolicyModel()
ref_model = SimplePolicyModel()  # 参考模型冻结

# 创建偏好数据
dpo_dataset = PreferenceDataset()
dpo_dataset.add_item(
    prompt="什么是机器学习？",
    chosen="机器学习是人工智能的一个分支，它使计算机能够从数据中学习模式，从而做出预测或决策，无需显式编程。",
    rejected="机器学习就是让机器学习。"
)
dpo_dataset.add_item(
    prompt="如何提高编程能力？",
    chosen="建议通过以下方式：1. 多做项目实践；2. 阅读优秀代码；3. 参与开源项目；4. 持续学习新技术。",
    rejected="多看书。"
)

# DPO 训练
dpo_trainer = DPOTrainer(policy_model, ref_model, beta=0.1)
dpo_trainer.train(dpo_dataset, epochs=5)

print("\n✅ DPO 训练完成!")
```

### 3.3 使用 Hugging Face TRL

```python
"""
使用 Hugging Face TRL 库进行 DPO 训练
"""

# 安装依赖
# pip install trl transformers peft datasets

from trl import DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def train_dpo_with_trl():
    """使用 TRL 进行 DPO 训练"""
    
    # 加载模型
    model_name = "Qwen/Qwen2-1.5B"
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 加载偏好数据
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:1000]")
    
    # DPO 训练参数
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        output_dir="./dpo_output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=5e-7,
        logging_steps=100,
    )
    
    # 创建 DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        beta=0.1,
    )
    
    # 训练
    dpo_trainer.train()
    
    # 保存
    dpo_trainer.save_model("./dpo_final")

# 实际使用时取消注释
# train_dpo_with_trl()
```

---

## 4. 对齐效果评估

```python
"""
对齐模型评估指标
"""

class AlignmentEvaluator:
    """对齐效果评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_instruction_following(self, model, test_cases):
        """评估指令遵循能力"""
        scores = []
        
        for case in test_cases:
            response = model.generate(case['instruction'])
            
            # 检查是否遵循指令
            # 实际中可以使用 LLM 作为评判
            score = self.check_instruction_followed(case, response)
            scores.append(score)
        
        return np.mean(scores)
    
    def evaluate_safety(self, model, harmful_prompts):
        """评估安全性"""
        refusal_count = 0
        
        for prompt in harmful_prompts:
            response = model.generate(prompt)
            
            if self.is_refusal(response):
                refusal_count += 1
        
        return refusal_count / len(harmful_prompts)
    
    def evaluate_helpfulness(self, model, test_cases):
        """评估有用性"""
        scores = []
        
        for case in test_cases:
            response = model.generate(case['prompt'])
            
            # 使用 GPT-4 或人工评分
            score = self.score_helpfulness(case, response)
            scores.append(score)
        
        return np.mean(scores)
    
    def check_instruction_followed(self, case, response):
        """检查指令是否被遵循（简化）"""
        return 1.0  # 实际使用 LLM 评判
    
    def is_refusal(self, response):
        """检查是否拒绝回答"""
        refusal_phrases = ["我不能", "抱歉", "无法", "I cannot", "I'm sorry"]
        return any(phrase in response for phrase in refusal_phrases)
    
    def score_helpfulness(self, case, response):
        """评分有用性（简化）"""
        return np.random.uniform(0, 1)  # 实际使用人工或 LLM 评判


# 对比不同对齐方法
print("\n对齐方法对比:")
print("-" * 50)
print(f"{'方法':<15} {'参数量':<10} {'训练复杂度':<15} {'效果'}")
print("-" * 50)
print(f"{'SFT':<15} {'全参数':<10} {'低':<15} {'基础对齐'}")
print(f"{'SFT + LoRA':<15} {'0.1%':<10} {'低':<15} {'基础对齐'}")
print(f"{'RLHF':<15} {'全参数':<10} {'高':<15} {'高级对齐'}")
print(f"{'DPO':<15} {'全参数':<10} {'中':<15} {'高级对齐'}")
print(f"{'DPO + LoRA':<15} {'0.1%':<10} {'低':<15} {'高级对齐'}")
```

---

## 📝 本章小结

### 核心要点

1. **SFT**：使用指令数据微调，让模型学会遵循指令
2. **RLHF**：三阶段训练（SFT → RM → PPO），对齐人类偏好
3. **DPO**：简化 RLHF，直接从偏好数据优化，无需奖励模型
4. **数据质量**：高质量的指令/偏好数据是对齐的关键

### 方法选择指南

| 方法 | 数据需求 | 计算资源 | 对齐效果 | 推荐场景 |
|------|----------|----------|----------|----------|
| SFT | 指令数据 | 低 | 基础 | 预算有限 |
| RLHF | 偏好数据 | 高 | 优秀 | 顶级产品 |
| DPO | 偏好数据 | 中 | 优秀 | 开源首选 |

---

<div align="center">

[⬅️ 上一章](../chapter17-peft/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter19-deployment/README.md)]

</div>
