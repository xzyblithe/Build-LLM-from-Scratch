# 第5章：词向量与文本表示

<div align="center">

[⬅️ 上一章](../chapter04-neural-networks/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter06-rnn/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解词向量的核心思想与意义
- ✅ 掌握 One-Hot 编码及其局限性
- ✅ 深入理解 Word2Vec（CBOW 和 Skip-gram）
- ✅ 掌握 GloVe 全局向量表示
- ✅ 了解 FastText 子词嵌入
- ✅ 从零实现 Word2Vec 模型
- ✅ 理解词向量的评估方法

---

## 🎯 本章内容

### 1. 文本表示概述

#### 1.1 为什么需要文本表示？

计算机无法直接理解文本，需要将文本转换为数值形式。

```
文本 → [表示方法] → 数值向量 → [模型] → 输出

核心问题：如何将离散的符号（词）转换为连续的向量？
```

#### 1.2 文本表示的演进

```
1. One-Hot 编码（独热编码）
   - 简单直观
   - 维度灾难
   - 无法表达语义关系

2. 分布式表示（Distributed Representation）
   - 低维稠密向量
   - 语义信息丰富
   - 词之间的相似度可计算

3. 上下文相关表示（Contextualized Representation）
   - BERT、GPT 等
   - 同一词在不同上下文有不同表示
   - 第8章详细介绍
```

---

### 2. One-Hot 编码

#### 2.1 基本原理

One-Hot 编码将每个词表示为一个独热向量，向量维度等于词汇表大小。

```
词汇表：["我", "喜欢", "学习", "AI"]

词 → One-Hot 向量：
"我"     → [1, 0, 0, 0]
"喜欢"   → [0, 1, 0, 0]
"学习"   → [0, 0, 1, 0]
"AI"     → [0, 0, 0, 1]
```

#### 2.2 代码实现

```python
import numpy as np

class OneHotEncoder:
    """One-Hot 编码器"""
    
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
    
    def fit(self, documents):
        """
        构建词汇表
        
        参数:
            documents: 文档列表，每个文档是词的列表
        """
        # 收集所有词
        word_set = set()
        for doc in documents:
            word_set.update(doc)
        
        # 构建映射
        self.vocab = {word: idx for idx, word in enumerate(sorted(word_set))}
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
        
        print(f"词汇表大小: {len(self.vocab)}")
    
    def encode(self, word):
        """
        将单个词编码为 One-Hot 向量
        
        参数:
            word: 单个词
        
        返回:
            one-hot 向量
        """
        vector = np.zeros(len(self.vocab))
        if word in self.vocab:
            vector[self.vocab[word]] = 1
        return vector
    
    def encode_sentence(self, sentence):
        """
        将句子编码为 One-Hot 矩阵
        
        参数:
            sentence: 词列表
        
        返回:
            shape: (句子长度, 词汇表大小)
        """
        return np.array([self.encode(word) for word in sentence])
    
    def decode(self, vector):
        """
        将 One-Hot 向量解码为词
        """
        idx = np.argmax(vector)
        return self.inv_vocab.get(idx, "<UNK>")

# 示例
documents = [
    ["我", "喜欢", "学习", "AI"],
    ["AI", "是", "未来", "的", "技术"],
    ["学习", "AI", "很有", "趣"]
]

encoder = OneHotEncoder()
encoder.fit(documents)

# 编码
print("\n'AI' 的 One-Hot 向量:")
print(encoder.encode("AI"))

print("\n句子编码:")
sentence = ["我", "喜欢", "AI"]
encoded = encoder.encode_sentence(sentence)
print(f"形状: {encoded.shape}")
print(encoded)
```

#### 2.3 One-Hot 的局限性

```
问题 1：维度灾难
- 词汇表大小通常为 10万~100万
- 每个词需要这么长的向量
- 存储和计算成本极高

问题 2：稀疏性
- 每个向量只有一个 1，其余都是 0
- 信息密度极低

问题 3：无法表达语义关系
- 任意两个不同词的向量正交（点积为 0）
- "好"和"优秀"的距离 = "好"和"坏"的距离
- 无法捕捉词之间的相似度

示例：
similarity("好", "优秀") = 0
similarity("好", "坏") = 0
（两者都是 0，无法区分）
```

---

### 3. 分布式表示的核心思想

#### 3.1 分布式假设

**"一个词的含义由它周围出现的词决定。"** —— J.R. Firth (1957)

```
语境相同的词，语义相近：

句子 1: "我喜欢吃苹果"
句子 2: "他喜欢吃香蕉"

"苹果"和"香蕉"都出现在"喜欢吃"后面，
因此它们语义相近（都是水果）。
```

#### 3.2 词向量空间

将每个词映射到低维稠密向量空间，相似词在空间中距离较近。

```
词向量特性：

        国王 (King)
          ↑
          |
    王后 ←→ 男性
  (Queen)   (Man)
          |
          ↓
        女人 (Woman)

类比关系：
King - Man + Woman ≈ Queen
```

---

### 4. Word2Vec

Word2Vec 是 Google 在 2013 年提出的词向量训练方法，包含两种模型：

#### 4.1 CBOW（Continuous Bag of Words）

**核心思想**：根据上下文预测中心词

```
输入：上下文词（周围词）
输出：中心词（目标词）

示例：
句子："我 喜欢 学习 AI"

窗口大小 = 2 时：
上下文: ["我", "学习"] → 预测: "喜欢"
上下文: ["喜欢", "AI"] → 预测: "学习"
```

```
CBOW 结构：

上下文词: w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}
                ↓          ↓         ↓         ↓
             [Embedding] [Embedding] [Embedding] [Embedding]
                ↓          ↓         ↓         ↓
                └──────────┴─────────┴─────────┘
                              ↓
                         求和/平均
                              ↓
                        [Linear + Softmax]
                              ↓
                         预测中心词
```

#### 4.2 Skip-gram

**核心思想**：根据中心词预测上下文

```
输入：中心词（目标词）
输出：上下文词（周围词）

示例：
句子："我 喜欢 学习 AI"

窗口大小 = 2 时：
中心词: "喜欢" → 预测: ["我", "学习"]
中心词: "学习" → 预测: ["喜欢", "AI"]
```

```
Skip-gram 结构：

中心词: w_t
    ↓
[Embedding]
    ↓
[Linear + Softmax] × 4（每个位置一个）
    ↓              ↓              ↓              ↓
w_{t-2}        w_{t-1}        w_{t+1}        w_{t+2}
```

#### 4.3 两种模型对比

| 特性 | CBOW | Skip-gram |
|------|------|-----------|
| 输入 | 上下文词 | 中心词 |
| 输出 | 中心词 | 上下文词 |
| 训练速度 | 快 | 慢 |
| 低频词效果 | 一般 | 好 |
| 适用场景 | 大规模数据 | 小规模数据/低频词 |

#### 4.4 从零实现 Skip-gram

```python
import numpy as np
from collections import Counter

class SkipGram:
    """从零实现 Skip-gram Word2Vec"""
    
    def __init__(self, embedding_dim=100, window_size=2, 
                 min_count=5, learning_rate=0.025):
        """
        参数:
            embedding_dim: 词向量维度
            window_size: 上下文窗口大小
            min_count: 最小词频
            learning_rate: 学习率
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.learning_rate = learning_rate
        
        # 词到索引的映射
        self.word2idx = {}
        self.idx2word = {}
        
        # 词向量矩阵
        self.W_in = None   # 输入词向量（中心词）
        self.W_out = None  # 输出词向量（上下文词）
    
    def build_vocab(self, sentences):
        """
        构建词汇表
        
        参数:
            sentences: 句子列表，每个句子是词列表
        """
        # 统计词频
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)
        
        # 过滤低频词
        vocab = [word for word, count in word_counts.items() 
                 if count >= self.min_count]
        
        # 构建映射
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(vocab)
        
        # 保存词频（用于负采样）
        self.word_counts = {word: word_counts[word] for word in vocab}
        
        print(f"词汇表大小: {self.vocab_size}")
        
        # 初始化词向量
        self._init_embeddings()
    
    def _init_embeddings(self):
        """初始化词向量矩阵"""
        # Xavier 初始化
        self.W_in = np.random.randn(self.vocab_size, self.embedding_dim) * \
                    np.sqrt(2.0 / (self.vocab_size + self.embedding_dim))
        self.W_out = np.random.randn(self.vocab_size, self.embedding_dim) * \
                     np.sqrt(2.0 / (self.vocab_size + self.embedding_dim))
    
    def generate_training_data(self, sentences):
        """
        生成训练数据
        
        参数:
            sentences: 句子列表
        
        返回:
            (中心词索引, 上下文词索引) 列表
        """
        training_data = []
        
        for sentence in sentences:
            # 将词转换为索引
            indices = [self.word2idx[word] for word in sentence 
                      if word in self.word2idx]
            
            # 生成 (中心词, 上下文词) 对
            for i, center_idx in enumerate(indices):
                # 确定窗口范围
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                
                # 获取上下文词
                for j in range(start, end):
                    if j != i:  # 排除中心词本身
                        context_idx = indices[j]
                        training_data.append((center_idx, context_idx))
        
        return training_data
    
    def softmax(self, x):
        """Softmax 函数"""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    
    def forward(self, center_idx, context_idx):
        """
        前向传播
        
        参数:
            center_idx: 中心词索引
            context_idx: 正样本上下文词索引
        
        返回:
            loss: 损失值
            grad_in: 输入向量梯度
            grad_out: 输出向量梯度
        """
        # 获取中心词向量
        h = self.W_in[center_idx]  # (embedding_dim,)
        
        # 计算输出分数
        u = np.dot(self.W_out, h)  # (vocab_size,)
        
        # Softmax 概率
        y_pred = self.softmax(u)
        
        # 计算损失（交叉熵）
        loss = -np.log(y_pred[context_idx] + 1e-10)
        
        # 反向传播
        # 输出层梯度
        dy = y_pred.copy()
        dy[context_idx] -= 1  # softmax + cross-entropy 的梯度
        
        # 输出向量梯度
        grad_out = np.outer(dy, h)
        
        # 输入向量梯度
        grad_in = np.dot(dy, self.W_out)
        
        return loss, grad_in, grad_out
    
    def train_step(self, center_idx, context_idx):
        """单步训练"""
        loss, grad_in, grad_out = self.forward(center_idx, context_idx)
        
        # 更新参数
        self.W_in[center_idx] -= self.learning_rate * grad_in
        self.W_out -= self.learning_rate * grad_out
        
        return loss
    
    def train(self, sentences, epochs=5):
        """
        训练模型
        
        参数:
            sentences: 句子列表
            epochs: 训练轮数
        """
        # 生成训练数据
        training_data = self.generate_training_data(sentences)
        print(f"训练样本数: {len(training_data)}")
        
        total_loss = 0
        for epoch in range(epochs):
            epoch_loss = 0
            
            # 打乱数据
            np.random.shuffle(training_data)
            
            for center_idx, context_idx in training_data:
                loss = self.train_step(center_idx, context_idx)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(training_data)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def get_embedding(self, word):
        """获取词向量"""
        if word not in self.word2idx:
            return None
        return self.W_in[self.word2idx[word]]
    
    def most_similar(self, word, top_k=5):
        """
        找最相似的词
        
        参数:
            word: 查询词
            top_k: 返回数量
        
        返回:
            [(词, 相似度), ...]
        """
        if word not in self.word2idx:
            return []
        
        # 获取查询词向量
        word_vec = self.W_in[self.word2idx[word]]
        
        # 计算与所有词的余弦相似度
        norms = np.linalg.norm(self.W_in, axis=1)
        word_norm = np.linalg.norm(word_vec)
        
        similarities = np.dot(self.W_in, word_vec) / (norms * word_norm + 1e-10)
        
        # 获取 top_k（排除自身）
        most_similar_idx = np.argsort(similarities)[::-1][1:top_k+1]
        
        return [(self.idx2word[idx], similarities[idx]) 
                for idx in most_similar_idx]


# 示例：训练 Skip-gram
sentences = [
    "我 喜欢 学习 机器学习".split(),
    "机器学习 是 AI 的 子领域".split(),
    "深度学习 是 机器学习 的 分支".split(),
    "神经网络 是 深度学习 的 基础".split(),
    "我 喜欢 AI 和 深度学习".split(),
    "学习 AI 需要 数学 基础".split(),
    "机器学习 和 深度学习 很 有趣".split(),
]

# 创建并训练模型
model = SkipGram(embedding_dim=50, window_size=2, min_count=1, 
                 learning_rate=0.1)
model.build_vocab(sentences)
model.train(sentences, epochs=100)

# 测试相似词
print("\n=== 相似词测试 ===")
test_words = ["机器学习", "AI", "学习"]
for word in test_words:
    if word in model.word2idx:
        print(f"\n'{word}' 最相似的词:")
        for sim_word, score in model.most_similar(word, top_k=3):
            print(f"  {sim_word}: {score:.4f}")
```

#### 4.5 负采样优化

上述实现的 Skip-gram 使用完整的 softmax，计算量大。实际应用中使用**负采样（Negative Sampling）**优化。

```python
class SkipGramNegSampling:
    """带负采样的 Skip-gram"""
    
    def __init__(self, vocab_size, embedding_dim, neg_samples=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.neg_samples = neg_samples
        
        # 初始化词向量
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.01
    
    def sigmoid(self, x):
        """Sigmoid 函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sample_negatives(self, word_counts, num_samples):
        """
        负采样
        
        参数:
            word_counts: 词频字典
            num_samples: 采样数量
        """
        # 计算采样概率（词频的 3/4 次方）
        total = sum(count ** 0.75 for count in word_counts.values())
        probs = np.array([count ** 0.75 / total 
                         for count in word_counts.values()])
        
        return np.random.choice(self.vocab_size, size=num_samples, p=probs)
    
    def train_step(self, center_idx, context_idx, neg_indices):
        """
        单步训练（带负采样）
        
        目标：最大化正样本概率，最小化负样本概率
        """
        # 获取向量
        h = self.W_in[center_idx]
        context_vec = self.W_out[context_idx]
        
        # 正样本：希望 sigmoid(h · v_context) 接近 1
        score = np.dot(h, context_vec)
        pos_loss = -np.log(self.sigmoid(score) + 1e-10)
        
        # 正样本梯度
        pos_grad = (self.sigmoid(score) - 1) * h
        h_grad = (self.sigmoid(score) - 1) * context_vec
        
        # 负样本：希望 sigmoid(h · v_neg) 接近 0
        neg_loss = 0
        for neg_idx in neg_indices:
            neg_vec = self.W_out[neg_idx]
            score = np.dot(h, neg_vec)
            neg_loss -= np.log(self.sigmoid(-score) + 1e-10)
            
            # 负样本梯度
            neg_grad = self.sigmoid(score) * h
            h_grad += self.sigmoid(score) * neg_vec
            
            # 更新负样本向量
            self.W_out[neg_idx] -= 0.025 * neg_grad
        
        # 更新参数
        self.W_in[center_idx] -= 0.025 * h_grad
        self.W_out[context_idx] -= 0.025 * pos_grad
        
        return pos_loss + neg_loss
```

---

### 5. GloVe（Global Vectors）

#### 5.1 核心思想

GloVe（2014）结合了全局矩阵分解和局部上下文窗口的优点。

```
Word2Vec: 基于局部上下文窗口
GloVe: 基于全局词共现矩阵

核心思想：利用词共现统计信息
```

#### 5.2 词共现矩阵

```
假设语料库：
"I like NLP"
"I like deep learning"
"deep learning is fun"

窗口大小 = 1 的共现矩阵：

        I  like  NLP  deep  learning  is  fun
I       0    2    1    0      0       0    0
like    2    0    1    1      1       0    0
NLP     1    1    0    0      0       0    0
deep    0    1    0    0      2       0    0
learning 0   1    0    2      0       1   0
is      0    0    0    0      1       0   1
fun     0    0    0    0      0       1   0

X_{ij} = 词 i 和词 j 在窗口内共现的次数
```

#### 5.3 GloVe 目标函数

```
目标函数：
J = Σ_{i,j} f(X_{ij}) (w_i · w̃_j + b_i + b̃_j - log X_{ij})²

其中：
- w_i: 词 i 的词向量
- w̃_j: 词 j 的上下文向量
- b_i, b̃_j: 偏置项
- f(X_{ij}): 权重函数，减少高频词的影响

权重函数：
f(x) = (x / x_max)^α  if x < x_max
f(x) = 1              otherwise

通常 x_max = 100, α = 0.75
```

#### 5.4 GloVe 实现

```python
import numpy as np
from collections import defaultdict

class GloVe:
    """简化版 GloVe 实现"""
    
    def __init__(self, embedding_dim=100, window_size=5, 
                 x_max=100, alpha=0.75, learning_rate=0.05):
        """
        参数:
            embedding_dim: 词向量维度
            window_size: 上下文窗口大小
            x_max: 权重函数截断值
            alpha: 权重函数指数
            learning_rate: 学习率
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.x_max = x_max
        self.alpha = alpha
        self.learning_rate = learning_rate
        
        self.word2idx = {}
        self.idx2word = {}
        self.W = None      # 词向量
        self.W_context = None  # 上下文向量
        self.b = None      # 词偏置
        self.b_context = None  # 上下文偏置
    
    def build_vocab(self, sentences):
        """构建词汇表"""
        word_set = set()
        for sentence in sentences:
            word_set.update(sentence)
        
        self.word2idx = {word: idx for idx, word in enumerate(word_set)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(word_set)
        
        # 初始化参数
        self.W = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
        self.W_context = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
        self.b = np.zeros(self.vocab_size)
        self.b_context = np.zeros(self.vocab_size)
        
        print(f"词汇表大小: {self.vocab_size}")
    
    def build_cooccurrence_matrix(self, sentences):
        """
        构建共现矩阵
        
        返回:
            共现字典: {(i, j): count}
        """
        cooccur = defaultdict(float)
        
        for sentence in sentences:
            indices = [self.word2idx[w] for w in sentence]
            
            for i, center_idx in enumerate(indices):
                # 窗口内所有位置
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if j != i:
                        context_idx = indices[j]
                        # 距离加权（越近权重越高）
                        distance = abs(i - j)
                        cooccur[(center_idx, context_idx)] += 1.0 / distance
        
        return cooccur
    
    def weight_function(self, x):
        """权重函数"""
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        return 1.0
    
    def train(self, sentences, epochs=50):
        """训练 GloVe"""
        # 构建共现矩阵
        cooccur = self.build_cooccurrence_matrix(sentences)
        print(f"非零共现数: {len(cooccur)}")
        
        # 转换为列表便于迭代
        cooccur_list = list(cooccur.items())
        
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(cooccur_list)
            
            for (i, j), x_ij in cooccur_list:
                # 计算权重
                weight = self.weight_function(x_ij)
                
                # 计算损失
                diff = np.dot(self.W[i], self.W_context[j]) + \
                       self.b[i] + self.b_context[j] - np.log(x_ij)
                
                loss = weight * diff ** 2
                total_loss += loss
                
                # 计算梯度
                grad_W_i = weight * diff * self.W_context[j]
                grad_W_context_j = weight * diff * self.W[i]
                grad_b_i = weight * diff
                grad_b_context_j = weight * diff
                
                # 更新参数
                self.W[i] -= self.learning_rate * grad_W_i
                self.W_context[j] -= self.learning_rate * grad_W_context_j
                self.b[i] -= self.learning_rate * grad_b_i
                self.b_context[j] -= self.learning_rate * grad_b_context_j
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.2f}")
    
    def get_embedding(self, word):
        """获取词向量（使用两个矩阵的平均）"""
        if word not in self.word2idx:
            return None
        idx = self.word2idx[word]
        return (self.W[idx] + self.W_context[idx]) / 2
    
    def most_similar(self, word, top_k=5):
        """找最相似的词"""
        if word not in self.word2idx:
            return []
        
        word_vec = self.get_embedding(word)
        
        # 计算相似度
        embeddings = (self.W + self.W_context) / 2
        norms = np.linalg.norm(embeddings, axis=1)
        word_norm = np.linalg.norm(word_vec)
        
        similarities = np.dot(embeddings, word_vec) / (norms * word_norm + 1e-10)
        
        # 排序
        most_similar_idx = np.argsort(similarities)[::-1][1:top_k+1]
        
        return [(self.idx2word[idx], similarities[idx]) 
                for idx in most_similar_idx]


# 示例
sentences = [
    "我 喜欢 学习 机器学习".split(),
    "机器学习 是 AI 的 子领域".split(),
    "深度学习 是 机器学习 的 分支".split(),
    "神经网络 是 深度学习 的 基础".split(),
]

glove = GloVe(embedding_dim=50, window_size=2)
glove.build_vocab(sentences)
glove.train(sentences, epochs=100)

print("\n=== 相似词测试 ===")
for word in ["机器学习", "学习"]:
    if word in glove.word2idx:
        print(f"\n'{word}' 最相似的词:")
        for sim_word, score in glove.most_similar(word, top_k=3):
            print(f"  {sim_word}: {score:.4f}")
```

---

### 6. FastText

#### 6.1 核心思想

FastText（Facebook, 2016）在 Word2Vec 基础上引入了**子词（Subword）**信息。

```
Word2Vec 问题：
- 无法处理未登录词（OOV）
- 无法利用词的内部结构

FastText 解决方案：
- 将词分解为字符 n-gram
- 词向量 = 所有子词向量的和
```

#### 6.2 字符 n-gram

```
词: "apple"
n = 3 (trigram)

子词: "<ap", "app", "ppl", "ple", "le>"
      (< 和 > 是边界标记)

词向量 = v("<ap") + v("app") + v("ppl") + v("ple") + v("le>") + v("<apple>")
```

#### 6.3 FastText 优势

```
1. 处理未登录词
   - 新词 "apples" 可以通过子词向量计算得到表示

2. 捕捉词形信息
   - "running", "runs", "ran" 共享子词，有相似向量

3. 更好的小语种支持
   - 词形丰富的语言（德语、俄语）效果更好
```

---

### 7. 词向量评估

#### 7.1 内在评估

**词相似度任务**：计算词向量相似度与人工标注的相关性。

```python
from scipy.stats import spearmanr

def evaluate_word_similarity(embeddings, word_pairs, human_scores):
    """
    评估词向量在词相似度任务上的表现
    
    参数:
        embeddings: 词向量字典 {word: vector}
        word_pairs: 词对列表 [(word1, word2), ...]
        human_scores: 人工标注的相似度分数
    """
    model_scores = []
    
    for (w1, w2) in word_pairs:
        if w1 in embeddings and w2 in embeddings:
            v1, v2 = embeddings[w1], embeddings[w2]
            # 余弦相似度
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            model_scores.append(sim)
        else:
            model_scores.append(0)
    
    # 计算 Spearman 相关系数
    correlation, _ = spearmanr(model_scores, human_scores)
    return correlation

# 示例
word_pairs = [
    ("king", "queen"),
    ("man", "woman"),
    ("good", "bad"),
]
human_scores = [0.8, 0.9, 0.3]  # 人工标注的相似度
```

#### 7.2 词类比任务

```
经典类比:
King - Man + Woman = Queen

计算方法:
1. 找到最接近 v("King") - v("Man") + v("Woman") 的词
2. 排除输入词本身
```

```python
def word_analogy(embeddings, word2idx, idx2word, 
                 word_a, word_b, word_c, top_k=5):
    """
    词类比: A is to B as C is to ?
    
    计算: vec_B - vec_A + vec_C
    """
    W = np.array([embeddings[w] for w in word2idx.keys()])
    
    vec_a = embeddings[word_a]
    vec_b = embeddings[word_b]
    vec_c = embeddings[word_c]
    
    # 目标向量
    target = vec_b - vec_a + vec_c
    
    # 计算相似度
    norms = np.linalg.norm(W, axis=1)
    target_norm = np.linalg.norm(target)
    similarities = np.dot(W, target) / (norms * target_norm + 1e-10)
    
    # 排除输入词
    exclude = {word_a, word_b, word_c}
    for word in exclude:
        if word in word2idx:
            similarities[word2idx[word]] = -np.inf
    
    # 返回 top_k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    return [(idx2word[idx], similarities[idx]) for idx in top_indices]

# 示例
# result = word_analogy(embeddings, "king", "man", "woman")
# print("'king' - 'man' + 'woman' =", result[0][0])
```

---

## 💻 完整代码示例

### 示例：训练中文词向量

```python
"""
完整示例：训练中文词向量
使用维基百科语料（简化版示例）
"""
import numpy as np
from collections import Counter, defaultdict
import re

class Word2VecTrainer:
    """完整的 Word2Vec 训练器"""
    
    def __init__(self, embedding_dim=100, window_size=5, 
                 min_count=5, neg_samples=5, learning_rate=0.025):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.neg_samples = neg_samples
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
    
    def preprocess(self, text):
        """文本预处理"""
        # 简单分词（实际应用使用 jieba 等）
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text.lower())
        return words
    
    def build_vocab(self, corpus):
        """构建词汇表"""
        word_counts = Counter()
        for text in corpus:
            words = self.preprocess(text)
            word_counts.update(words)
        
        # 过滤低频词
        vocab = {word: idx for idx, (word, count) in enumerate(
            word_counts.most_common()) if count >= self.min_count}
        
        self.word2idx = vocab
        self.idx2word = {idx: word for word, idx in vocab.items()}
        self.vocab_size = len(vocab)
        self.word_counts = {word: word_counts[word] for word in vocab}
        
        # 计算负采样概率
        total = sum(count ** 0.75 for count in self.word_counts.values())
        self.neg_sampling_probs = np.array([
            count ** 0.75 / total for count in self.word_counts.values()
        ])
        
        print(f"词汇表大小: {self.vocab_size}")
        
        # 初始化词向量
        self.W_in = (np.random.rand(self.vocab_size, self.embedding_dim) - 0.5) / \
                    self.embedding_dim
        self.W_out = np.zeros((self.vocab_size, self.embedding_dim))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    def train(self, corpus, epochs=5):
        """训练模型"""
        # 生成训练数据
        training_data = []
        for text in corpus:
            words = self.preprocess(text)
            indices = [self.word2idx[w] for w in words if w in self.word2idx]
            
            for i, center in enumerate(indices):
                # 动态窗口
                window = np.random.randint(1, self.window_size + 1)
                start = max(0, i - window)
                end = min(len(indices), i + window + 1)
                
                for j in range(start, end):
                    if j != i:
                        training_data.append((center, indices[j]))
        
        print(f"训练样本数: {len(training_data)}")
        
        total_words = epochs * len(training_data)
        word_count = 0
        
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            epoch_loss = 0
            
            for center, context in training_data:
                # 负采样
                neg_indices = np.random.choice(
                    self.vocab_size, 
                    size=self.neg_samples, 
                    p=self.neg_sampling_probs
                )
                
                # 获取向量
                h = self.W_in[center]
                
                # 正样本更新
                score = np.dot(h, self.W_out[context])
                grad = (self.sigmoid(score) - 1)
                
                h_grad = grad * self.W_out[context]
                self.W_out[context] -= self.learning_rate * grad * h
                
                # 负样本更新
                for neg_idx in neg_indices:
                    score = np.dot(h, self.W_out[neg_idx])
                    grad = self.sigmoid(score)
                    
                    h_grad += grad * self.W_out[neg_idx]
                    self.W_out[neg_idx] -= self.learning_rate * grad * h
                
                # 更新中心词向量
                self.W_in[center] -= self.learning_rate * h_grad
                
                # 更新学习率
                word_count += 1
                self.learning_rate = self.initial_lr * (1 - word_count / total_words)
                self.learning_rate = max(self.learning_rate, self.initial_lr * 0.0001)
            
            print(f"Epoch {epoch + 1}/{epochs} 完成")
    
    def get_embedding(self, word):
        if word in self.word2idx:
            return self.W_in[self.word2idx[word]]
        return None
    
    def most_similar(self, word, top_k=10):
        if word not in self.word2idx:
            return []
        
        word_vec = self.W_in[self.word2idx[word]]
        norms = np.linalg.norm(self.W_in, axis=1)
        word_norm = np.linalg.norm(word_vec)
        
        similarities = np.dot(self.W_in, word_vec) / (norms * word_norm + 1e-10)
        most_similar_idx = np.argsort(similarities)[::-1][1:top_k+1]
        
        return [(self.idx2word[idx], similarities[idx]) for idx in most_similar_idx]


# 示例语料（实际应用使用更大规模语料）
corpus = [
    "机器学习是人工智能的一个重要分支",
    "深度学习是机器学习的一个子领域",
    "神经网络是深度学习的基础架构",
    "自然语言处理是人工智能的重要应用",
    "计算机视觉也是人工智能的重要方向",
    "机器学习包括监督学习和无监督学习",
    "深度学习使用多层神经网络进行学习",
    "自然语言处理可以用于文本分类和情感分析",
    "计算机视觉可以用于图像识别和目标检测",
    "人工智能技术正在快速发展",
    "机器学习算法需要大量数据训练",
    "深度学习模型需要强大的计算能力",
    "神经网络模拟人脑神经元的工作方式",
    "自然语言处理让计算机理解人类语言",
    "计算机视觉让计算机理解图像内容",
]

# 训练
trainer = Word2VecTrainer(embedding_dim=50, window_size=3, 
                          min_count=1, neg_samples=3)
trainer.build_vocab(corpus)
trainer.train(corpus, epochs=100)

# 测试
print("\n=== 词向量测试 ===")
test_words = ["机器学习", "深度学习", "人工智能", "神经网络"]
for word in test_words:
    if word in trainer.word2idx:
        print(f"\n'{word}' 最相似的词:")
        for sim_word, score in trainer.most_similar(word, top_k=5):
            print(f"  {sim_word}: {score:.4f}")

# 保存词向量
print("\n=== 保存词向量 ===")
with open("word_vectors.txt", "w", encoding="utf-8") as f:
    f.write(f"{trainer.vocab_size} {trainer.embedding_dim}\n")
    for word, idx in trainer.word2idx.items():
        vec = " ".join(map(str, trainer.W_in[idx]))
        f.write(f"{word} {vec}\n")
print("词向量已保存到 word_vectors.txt")
```

---

## 🎯 实践练习

### 练习 1：使用预训练词向量

**任务**：加载预训练的 GloVe 词向量，完成词类比任务。

```python
def load_glove_vectors(filepath):
    """加载 GloVe 词向量"""
    # TODO: 实现
    pass

# 加载后测试
# King - Man + Woman = ?
# Beijing - China + Japan = ?
```

### 练习 2：可视化词向量

**任务**：使用 t-SNE 或 PCA 降维可视化词向量。

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, words):
    """可视化词向量"""
    # TODO: 实现
    pass
```

### 练习 3：训练领域词向量

**任务**：在特定领域语料上训练词向量，观察与通用词向量的差异。

---

## 📝 本章小结

### 核心要点

1. **One-Hot 编码**：简单但存在维度灾难、稀疏性、无法表达语义等问题
2. **分布式表示**：低维稠密向量，能表达语义关系
3. **Word2Vec**：CBOW（上下文预测中心词）和 Skip-gram（中心词预测上下文）
4. **负采样**：优化 softmax 计算量大的问题
5. **GloVe**：利用全局共现统计，结合矩阵分解优点
6. **FastText**：引入子词信息，处理未登录词

### 关键公式

```
One-Hot: v(w) = [0, ..., 1, ..., 0]（仅一个位置为1）

Skip-gram 目标: max Σ log P(context | center)
P(w_c | w_t) = exp(v_c · v_t) / Σ exp(v_i · v_t)

GloVe 目标: min Σ f(X_ij) (w_i · w̃_j + b_i + b̃_j - log X_ij)²

词类比: v(B) - v(A) + v(C) ≈ v(D)
```

### 方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| One-Hot | 简单 | 维度灾难、无语义 | 小规模分类 |
| Word2Vec | 高效、效果好 | 不考虑全局信息 | 大规模语料 |
| GloVe | 全局统计 | 需构建共现矩阵 | 中小规模语料 |
| FastText | 处理未登录词 | 模型更大 | 词形丰富语言 |

---

<div align="center">

[⬅️ 上一章](../chapter04-neural-networks/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter06-rnn/README.md)

</div>
