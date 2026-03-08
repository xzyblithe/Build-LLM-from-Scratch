# 第9章：预训练语言模型

<div align="center">

[⬅️ 上一章](../chapter08-transformer/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter10-llm-principles/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解预训练语言模型的核心思想
- ✅ 掌握自编码（BERT）与自回归（GPT）的区别
- ✅ 深入理解 BERT 的训练目标（MLM + NSP）
- ✅ 理解 GPT 的因果语言模型训练
- ✅ 掌握预训练的关键技术
- ✅ 了解主流预训练模型的发展

---

## 🎯 本章内容

### 1. 预训练语言模型概述

#### 1.1 为什么需要预训练？

```
传统方法：
1. 收集任务特定数据
2. 从随机初始化开始训练
3. 数据量小时容易过拟合

问题：
- 标注数据昂贵
- 不同任务从头训练
- 无法利用通用语言知识
```

```
预训练 + 微调范式：
1. 在大规模无标注文本上预训练
2. 学习通用语言表示
3. 在下游任务上微调

优势：
- 利用海量无标注数据
- 学习通用语言知识
- 下游任务只需少量标注数据
```

#### 1.2 预训练范式对比

| 范式 | 代表模型 | 训练目标 | 特点 |
|------|---------|---------|------|
| **自编码** | BERT | 掩码语言模型 | 双向编码，适合理解任务 |
| **自回归** | GPT | 因果语言模型 | 单向编码，适合生成任务 |
| **编码器-解码器** | T5、BART | Seq2Seq | 兼顾理解与生成 |

---

### 2. BERT：双向编码器表示

#### 2.1 BERT 架构

```
BERT = Transformer Encoder

结构：
- 多层 Transformer 编码器
- 双向注意力（每个词能看到所有词）

两种规模：
BERT-Base: L=12, H=768, A=12, 参数=110M
BERT-Large: L=24, H=1024, A=16, 参数=340M

其中：
- L: 层数
- H: 隐藏维度
- A: 注意力头数
```

#### 2.2 输入表示

```
BERT 输入 = Token Embedding + Segment Embedding + Position Embedding

[CLS]  我  喜欢  AI  [SEP]  机器  学习  很  有趣  [SEP]
  ↓     ↓    ↓    ↓    ↓      ↓     ↓    ↓    ↓     ↓
Token Embedding

  0     0    0    0    0      1     1    1    1     1
  ↓     ↓    ↓    ↓    ↓      ↓     ↓    ↓    ↓     ↓
Segment Embedding（区分两个句子）

  0     1    2    3    4      5     6    7    8     9
  ↓     ↓    ↓    ↓    ↓      ↓     ↓    ↓    ↓     ↓
Position Embedding（可学习）
```

```python
import numpy as np

class BERTEmbedding:
    """BERT 输入嵌入"""
    
    def __init__(self, vocab_size, max_len, d_model, num_segments=2):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        
        # Token 嵌入
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # Segment 嵌入
        self.segment_embedding = np.random.randn(num_segments, d_model) * 0.02
        
        # Position 嵌入（可学习）
        self.position_embedding = np.random.randn(max_len, d_model) * 0.02
    
    def forward(self, token_ids, segment_ids):
        """
        计算输入嵌入
        
        参数:
            token_ids: 词索引, shape (seq_len,)
            segment_ids: 句子索引, shape (seq_len,)
        
        返回:
            embeddings: shape (seq_len, d_model)
        """
        seq_len = len(token_ids)
        
        # Token 嵌入
        token_emb = self.token_embedding[token_ids]
        
        # Segment 嵌入
        segment_emb = self.segment_embedding[segment_ids]
        
        # Position 嵌入
        position_emb = self.position_embedding[:seq_len]
        
        return token_emb + segment_emb + position_emb


# 测试
vocab_size = 1000
max_len = 512
d_model = 128

embedding = BERTEmbedding(vocab_size, max_len, d_model)

token_ids = np.array([101, 100, 200, 300, 102, 400, 500, 102])  # [CLS] 我 喜欢 AI [SEP] ...
segment_ids = np.array([0, 0, 0, 0, 0, 1, 1, 1])

output = embedding.forward(token_ids, segment_ids)
print("输入嵌入形状:", output.shape)
```

#### 2.3 掩码语言模型（MLM）

```
核心思想：随机掩盖输入词，预测被掩盖的词

步骤：
1. 随机选择 15% 的 token
2. 其中：
   - 80% 替换为 [MASK]
   - 10% 替换为随机词
   - 10% 保持不变
3. 预测被选中的 token

示例：
输入：我 喜欢 AI 和 机器学习
掩码：我 喜欢 [MASK] 和 机器学习
目标：预测 [MASK] = "AI"

为什么不全部用 [MASK]？
- 训练和推理不一致（推理时没有 [MASK]）
- 10% 随机词：学习纠错能力
- 10% 不变：保持信息流
```

```python
def create_mlm_mask(token_ids, vocab_size, mask_token_id, mask_prob=0.15):
    """
    创建 MLM 掩码
    
    参数:
        token_ids: 原始 token 序列
        vocab_size: 词汇表大小
        mask_token_id: [MASK] 的 ID
        mask_prob: 掩码概率
    
    返回:
        masked_tokens: 掩码后的 token
        masked_positions: 被掩码的位置
        masked_labels: 掩码位置的真实标签
    """
    token_ids = token_ids.copy()
    seq_len = len(token_ids)
    
    # 随机选择要掩码的位置
    mask_positions = []
    for i in range(seq_len):
        if np.random.random() < mask_prob:
            mask_positions.append(i)
    
    masked_labels = token_ids[mask_positions].copy()
    
    # 应用掩码策略
    for pos in mask_positions:
        prob = np.random.random()
        if prob < 0.8:
            # 80% 替换为 [MASK]
            token_ids[pos] = mask_token_id
        elif prob < 0.9:
            # 10% 替换为随机词
            token_ids[pos] = np.random.randint(0, vocab_size)
        # 10% 保持不变
    
    return token_ids, np.array(mask_positions), masked_labels


# 测试
np.random.seed(42)
token_ids = np.array([101, 100, 200, 300, 400, 500, 102])
vocab_size = 1000
mask_token_id = 103  # [MASK]

masked, positions, labels = create_mlm_mask(token_ids, vocab_size, mask_token_id)
print("原始:", token_ids)
print("掩码后:", masked)
print("掩码位置:", positions)
print("真实标签:", labels)
```

#### 2.4 下一句预测（NSP）

```
目标：判断两个句子是否连续

训练数据：
- 50% 正样本：实际连续的句子对
- 50% 负样本：随机配对的句子对

输入格式：
[CLS] 句子A [SEP] 句子B [SEP]

输出：
[CLS] 位置的向量 → 二分类 → 是否连续
```

```python
def create_nsp_data(sentence_pairs, cls_id=101, sep_id=102):
    """
    创建 NSP 训练数据
    
    参数:
        sentence_pairs: 句子对列表 [(sent_a, sent_b, is_next), ...]
        cls_id: [CLS] token ID
        sep_id: [SEP] token ID
    
    返回:
        token_ids: token 序列
        segment_ids: segment 标识
        labels: 是否连续
    """
    token_ids_list = []
    segment_ids_list = []
    labels = []
    
    for sent_a, sent_b, is_next in sentence_pairs:
        # 构建输入
        tokens = [cls_id] + sent_a + [sep_id] + sent_b + [sep_id]
        segments = [0] * (len(sent_a) + 2) + [1] * (len(sent_b) + 1)
        
        token_ids_list.append(tokens)
        segment_ids_list.append(segments)
        labels.append(is_next)
    
    return token_ids_list, segment_ids_list, np.array(labels)


# 示例
sentence_pairs = [
    ([100, 200], [300, 400], 1),  # 连续
    ([100, 200], [500, 600], 0),  # 不连续
]

token_ids, segment_ids, labels = create_nsp_data(sentence_pairs)
print("Token IDs:", token_ids)
print("Segment IDs:", segment_ids)
print("Labels:", labels)
```

#### 2.5 BERT 完整训练目标

```python
class BERTPretrainingHead:
    """BERT 预训练头"""
    
    def __init__(self, d_model, vocab_size):
        # MLM 头
        self.mlm_head = np.random.randn(d_model, vocab_size) * 0.02
        
        # NSP 头
        self.nsp_head = np.random.randn(d_model, 2) * 0.02
    
    def forward(self, encoder_output, masked_positions=None):
        """
        前向传播
        
        参数:
            encoder_output: 编码器输出, shape (seq_len, d_model)
            masked_positions: 掩码位置
        
        返回:
            mlm_logits: MLM 预测, shape (num_masked, vocab_size)
            nsp_logits: NSP 预测, shape (2,)
        """
        # NSP 预测（使用 [CLS] 即第一个位置）
        nsp_logits = np.dot(encoder_output[0], self.nsp_head)
        
        # MLM 预测
        if masked_positions is not None:
            mlm_logits = np.dot(encoder_output[masked_positions], self.mlm_head)
        else:
            mlm_logits = None
        
        return mlm_logits, nsp_logits


def compute_pretraining_loss(mlm_logits, mlm_labels, nsp_logits, nsp_labels):
    """计算预训练损失"""
    # MLM 损失（交叉熵）
    mlm_probs = np.exp(mlm_logits - np.max(mlm_logits, axis=1, keepdims=True))
    mlm_probs = mlm_probs / np.sum(mlm_probs, axis=1, keepdims=True)
    
    mlm_loss = 0
    for i, label in enumerate(mlm_labels):
        mlm_loss -= np.log(mlm_probs[i, label] + 1e-10)
    mlm_loss /= len(mlm_labels)
    
    # NSP 损失
    nsp_probs = np.exp(nsp_logits - np.max(nsp_logits))
    nsp_probs = nsp_probs / np.sum(nsp_probs)
    nsp_loss = -np.log(nsp_probs[nsp_labels] + 1e-10)
    
    return mlm_loss + nsp_loss
```

---

### 3. GPT：生成式预训练

#### 3.1 GPT 架构

```
GPT = Transformer Decoder（去掉 Cross-Attention）

结构：
- 多层 Transformer 解码器
- 因果掩码（只能看到之前的词）
- 单向注意力

规模演进：
GPT-1:  12层,  768维, 参数 117M
GPT-2:  48层, 1600维, 参数 1.5B
GPT-3:  96层,12288维, 参数 175B
```

#### 3.2 因果语言模型

```
目标：预测下一个词

P(text) = Π P(w_t | w_1, w_2, ..., w_{t-1})

示例：
输入：我 喜欢 学习
目标：预测"学习"之后的词
     可能是"AI"、"编程"、"数学"等

训练：
最大化似然：max Σ log P(w_t | w_{<t})
```

```python
class GPTModel:
    """简化的 GPT 模型"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token 嵌入
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # Position 嵌入
        self.position_embedding = np.random.randn(max_len, d_model) * 0.02
        
        # Transformer 解码器层（简化）
        self.layers = []
        for _ in range(num_layers):
            layer = {
                'W_q': np.random.randn(d_model, d_model) * 0.02,
                'W_k': np.random.randn(d_model, d_model) * 0.02,
                'W_v': np.random.randn(d_model, d_model) * 0.02,
                'W_o': np.random.randn(d_model, d_model) * 0.02,
                'W1': np.random.randn(d_model, d_model * 4) * 0.02,
                'W2': np.random.randn(d_model * 4, d_model) * 0.02,
            }
            self.layers.append(layer)
        
        # 输出层
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
    
    def causal_mask(self, seq_len):
        """创建因果掩码"""
        return np.triu(np.ones((seq_len, seq_len)), k=1)
    
    def attention(self, Q, K, V, mask):
        """带掩码的注意力"""
        d_k = Q.shape[-1]
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        scores += mask * -1e9
        
        weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        return np.dot(weights, V)
    
    def layer_norm(self, X):
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        return (X - mean) / np.sqrt(var + 1e-6)
    
    def forward(self, token_ids):
        """前向传播"""
        seq_len = len(token_ids)
        
        # 嵌入
        X = self.token_embedding[token_ids] + self.position_embedding[:seq_len]
        
        # 因果掩码
        mask = self.causal_mask(seq_len)
        
        # 解码器层
        for layer in self.layers:
            # Self-Attention
            Q = np.dot(X, layer['W_q'])
            K = np.dot(X, layer['W_k'])
            V = np.dot(X, layer['W_v'])
            attn_out = self.attention(Q, K, V, mask)
            attn_out = np.dot(attn_out, layer['W_o'])
            X = self.layer_norm(X + attn_out)
            
            # FFN
            ffn_out = np.maximum(0, np.dot(X, layer['W1']))
            ffn_out = np.dot(ffn_out, layer['W2'])
            X = self.layer_norm(X + ffn_out)
        
        # 输出 logits
        logits = np.dot(X, self.lm_head)
        
        return logits
    
    def generate(self, token_ids, max_new_tokens=50, temperature=1.0):
        """生成文本"""
        for _ in range(max_new_tokens):
            logits = self.forward(token_ids)
            
            # 只取最后一个位置的预测
            next_logits = logits[-1] / temperature
            
            # Softmax
            probs = np.exp(next_logits - np.max(next_logits))
            probs = probs / np.sum(probs)
            
            # 采样
            next_token = np.random.choice(self.vocab_size, p=probs)
            token_ids = np.append(token_ids, next_token)
        
        return token_ids


# 测试
np.random.seed(42)

gpt = GPTModel(vocab_size=1000, d_model=64, num_heads=4, num_layers=2, max_len=128)

input_ids = np.array([100, 200, 300])
logits = gpt.forward(input_ids)

print("输入:", input_ids)
print("输出形状:", logits.shape)
print("每个位置的预测（最高概率词）:", np.argmax(logits, axis=-1))
```

---

### 4. 其他预训练模型

#### 4.1 RoBERTa

```
改进点：
1. 更多数据（16GB → 160GB）
2. 更长训练（更多步数）
3. 更大批量
4. 去掉 NSP 任务
5. 动态掩码（每次迭代重新掩码）

结果：显著优于 BERT
```

#### 4.2 ALBERT

```
改进点：
1. 因式分解嵌入参数化
   - 将大嵌入矩阵分解为两个小矩阵
   
2. 跨层参数共享
   - 所有层使用相同参数
   
3. 句子顺序预测（SOP）
   - 替代 NSP，判断句子顺序
   
结果：参数减少，性能相当
```

#### 4.3 ELECTRA

```
核心思想：判别式预训练

训练方式：
1. 生成器（小型 BERT）替换部分词
2. 判别器判断每个词是否被替换

优势：
- 所有 token 都参与训练（而非只预测掩码词）
- 训练效率更高
```

#### 4.4 T5

```
核心思想：统一文本到文本框架

所有任务都转换为文本生成：
- 分类：输入文本 → 输出标签文本
- 翻译：输入语言 → 输出语言
- 问答：问题 → 答案

架构：编码器-解码器
```

---

### 5. 预训练技巧

#### 5.1 数据处理

```
1. 文本清洗
   - 去除乱码、重复文本
   - 过滤低质量内容

2. 分词
   - WordPiece（BERT）
   - BPE（GPT）
   - SentencePiece（T5）

3. 数据增强
   - 回译
   - 同义词替换
   - 随机删除/交换
```

#### 5.2 训练优化

```
1. 混合精度训练
   - FP16 计算，FP32 累加
   - 减少显存，加速训练

2. 梯度累积
   - 小批量多次前向传播
   - 模拟大批量训练

3. 分布式训练
   - 数据并行
   - 模型并行（大模型）

4. 预热（Warmup）
   - 学习率从 0 逐渐增加
   - 稳定训练初期
```

```python
def get_cosine_schedule(step, total_steps, warmup_steps, peak_lr):
    """余弦退火学习率调度"""
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return peak_lr * 0.5 * (1 + np.cos(np.pi * progress))


# 可视化
total_steps = 10000
warmup_steps = 1000
peak_lr = 1e-4

steps = range(total_steps)
lrs = [get_cosine_schedule(s, total_steps, warmup_steps, peak_lr) for s in steps]

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(steps, lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Cosine Learning Rate Schedule')
plt.grid(True, alpha=0.3)
# plt.show()
```

#### 5.3 高效微调

```
1. Adapter
   - 在每层插入小型适配器模块
   - 只训练适配器参数

2. LoRA
   - 低秩适配
   - 冻结原模型，训练低秩矩阵

3. Prefix Tuning
   - 在输入前添加可学习的前缀向量

4. Prompt Tuning
   - 只训练软提示嵌入
```

---

## 💻 完整代码示例

### 示例：简化版 BERT 预训练

```python
"""
完整示例：简化版 BERT 预训练
"""
import numpy as np

class SimpleBERT:
    """简化版 BERT"""
    
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 嵌入
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.segment_emb = np.random.randn(2, d_model) * 0.02
        self.position_emb = np.random.randn(512, d_model) * 0.02
        
        # 编码器层
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                'W_q': np.random.randn(d_model, d_model) * 0.02,
                'W_k': np.random.randn(d_model, d_model) * 0.02,
                'W_v': np.random.randn(d_model, d_model) * 0.02,
                'W_o': np.random.randn(d_model, d_model) * 0.02,
                'W1': np.random.randn(d_model, d_model * 4) * 0.02,
                'b1': np.zeros(d_model * 4),
                'W2': np.random.randn(d_model * 4, d_model) * 0.02,
                'b2': np.zeros(d_model),
            })
        
        # MLM 头
        self.mlm_head = np.random.randn(d_model, vocab_size) * 0.02
        
        # NSP 头
        self.nsp_head = np.random.randn(d_model, 2) * 0.02
    
    def attention(self, Q, K, V):
        d_k = Q.shape[-1]
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        return np.dot(weights, V)
    
    def layer_norm(self, X):
        mean = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        return (X - mean) / np.sqrt(var + 1e-6)
    
    def forward(self, token_ids, segment_ids):
        seq_len = len(token_ids)
        
        # 嵌入
        X = self.token_emb[token_ids] + self.segment_emb[segment_ids] + self.position_emb[:seq_len]
        
        # 编码器层
        for layer in self.layers:
            # Self-Attention
            Q = np.dot(X, layer['W_q'])
            K = np.dot(X, layer['W_k'])
            V = np.dot(X, layer['W_v'])
            attn_out = self.attention(Q, K, V)
            attn_out = np.dot(attn_out, layer['W_o'])
            X = self.layer_norm(X + attn_out)
            
            # FFN
            ffn_out = np.maximum(0, np.dot(X, layer['W1']) + layer['b1'])
            ffn_out = np.dot(ffn_out, layer['W2']) + layer['b2']
            X = self.layer_norm(X + ffn_out)
        
        return X
    
    def pretraining_forward(self, token_ids, segment_ids, masked_positions):
        """预训练前向传播"""
        encoder_output = self.forward(token_ids, segment_ids)
        
        # MLM 预测
        mlm_logits = np.dot(encoder_output[masked_positions], self.mlm_head)
        
        # NSP 预测
        nsp_logits = np.dot(encoder_output[0], self.nsp_head)
        
        return mlm_logits, nsp_logits
    
    def predict_masked_tokens(self, token_ids, segment_ids, masked_positions):
        """预测掩码词"""
        mlm_logits, _ = self.pretraining_forward(token_ids, segment_ids, masked_positions)
        return np.argmax(mlm_logits, axis=-1)


# 测试
np.random.seed(42)

vocab_size = 1000
model = SimpleBERT(vocab_size)

# 模拟输入
token_ids = np.array([101, 100, 200, 103, 400, 102, 500, 600, 102])  # 103 是 [MASK]
segment_ids = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
masked_positions = np.array([3])  # 第4个位置被掩码

# 预测
predictions = model.predict_masked_tokens(token_ids, segment_ids, masked_positions)
print("掩码位置预测:", predictions)
```

---

## 🎯 实践练习

### 练习 1：实现 WordPiece 分词

**任务**：实现 WordPiece 分词算法。

```python
def wordpiece_tokenize(word, vocab, max_length=100):
    """
    WordPiece 分词
    
    提示：
    1. 贪心匹配最长子词
    2. 不在词表中则拆分
    """
    # TODO: 实现
    pass
```

### 练习 2：实现 BPE 分词

**任务**：实现 Byte Pair Encoding 分词。

### 练习 3：对比预训练目标

**任务**：比较 MLM 和 CLM 在相同数据上的效果。

---

## 📝 本章小结

### 核心要点

1. **预训练范式**：自编码（BERT）、自回归（GPT）、编码器-解码器（T5）
2. **BERT**：MLM + NSP，双向编码，适合理解任务
3. **GPT**：因果语言模型，单向编码，适合生成任务
4. **预训练技巧**：数据处理、训练优化、高效微调

### 关键对比

| 模型 | 架构 | 预训练目标 | 优势 | 典型应用 |
|------|------|-----------|------|---------|
| BERT | Encoder | MLM + NSP | 双向理解 | 分类、NER |
| GPT | Decoder | CLM | 文本生成 | 生成任务 |
| T5 | Enc-Dec | Seq2Seq | 统一框架 | 多任务 |

### 发展趋势

```
规模增长：
BERT (2018): 340M 参数
GPT-2 (2019): 1.5B 参数
GPT-3 (2020): 175B 参数
GPT-4 (2023): 推测万亿级

技术演进：
- 更大规模
- 更好的分词
- 更高效的训练
- 多模态融合
```

---

<div align="center">

[⬅️ 上一章](../chapter08-transformer/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter10-llm-principles/README.md)

</div>
