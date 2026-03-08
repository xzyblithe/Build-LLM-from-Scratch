# 第5章：词向量与文本表示

<div align="center">

[⬅️ 上一章](../chapter04-neural-networks/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter06-rnn/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解文本如何转化为数值表示
- ✅ 掌握 One-Hot、词袋模型、TF-IDF 等方法
- ✅ 深入理解 Word2Vec 的原理和实现
- ✅ 掌握 GloVe 和 FastText
- ✅ 实现词向量的训练和应用

---

## 🎯 本章内容

### 1. 文本表示方法

#### 1.1 One-Hot 编码

最简单的文本表示方法，每个词用一个独热向量表示。

```python
import numpy as np

# 词汇表
vocab = ["我", "爱", "自然语言", "处理"]
vocab_size = len(vocab)

# One-Hot 编码
def one_hot_encode(word, vocab):
    """One-Hot 编码"""
    vector = np.zeros(len(vocab))
    if word in vocab:
        idx = vocab.index(word)
        vector[idx] = 1
    return vector

# 示例
for word in vocab:
    print(f"'{word}': {one_hot_encode(word, vocab)}")
```

**缺点**：
- 维度等于词汇表大小，稀疏
- 无法表达词与词之间的相似度
- 维度灾难

---

#### 1.2 词袋模型（Bag of Words）

统计每个词出现的次数。

```python
from collections import Counter

# 文档集合
documents = [
    "我 爱 自然语言 处理",
    "自然语言 处理 很 有趣",
    "我 爱 学习"
]

# 构建词汇表
vocab = set()
for doc in documents:
    words = doc.split()
    vocab.update(words)
vocab = list(vocab)

print(f"词汇表: {vocab}")

# 词袋表示
def bag_of_words(document, vocab):
    """词袋模型"""
    word_count = Counter(document.split())
    vector = [word_count.get(word, 0) for word in vocab]
    return vector

# 示例
for doc in documents:
    print(f"\n文档: '{doc}'")
    print(f"词袋向量: {bag_of_words(doc, vocab)}")
```

---

#### 1.3 TF-IDF

考虑词的重要性，平衡常见词和稀有词。

```python
import numpy as np
from collections import Counter

def compute_tf(doc):
    """计算词频 TF"""
    words = doc.split()
    word_count = Counter(words)
    tf = {word: count / len(words) for word, count in word_count.items()}
    return tf

def compute_idf(documents, vocab):
    """计算逆文档频率 IDF"""
    n_docs = len(documents)
    idf = {}
    
    for word in vocab:
        # 包含该词的文档数
        n_containing = sum(1 for doc in documents if word in doc.split())
        idf[word] = np.log(n_docs / (1 + n_containing))
    
    return idf

def compute_tfidf(doc, vocab, idf):
    """计算 TF-IDF"""
    tf = compute_tf(doc)
    tfidf = [tf.get(word, 0) * idf[word] for word in vocab]
    return np.array(tfidf)

# 示例
documents = [
    "我 爱 自然语言 处理",
    "自然语言 处理 很 有趣",
    "我 爱 学习"
]

vocab = list(set(word for doc in documents for word in doc.split()))
idf = compute_idf(documents, vocab)

print("TF-IDF 向量:")
for doc in documents:
    tfidf = compute_tfidf(doc, vocab, idf)
    print(f"'{doc}': {tfidf.round(3)}")
```

---

### 2. Word2Vec

Word2Vec 是词向量里程碑式的工作，包含两种模型：Skip-gram 和 CBOW。

#### 2.1 Skip-gram 模型

给定中心词，预测上下文词。

```
输入: 中心词
输出: 上下文词

例如："我 爱 自然语言 处理"
中心词 "爱" -> 预测 ["我", "自然语言"]
```

```python
import numpy as np

class SkipGram:
    """Skip-gram 模型"""
    
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        
        # 初始化权重
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_out = np.random.randn(embedding_dim, vocab_size) * 0.01
    
    def softmax(self, x):
        """Softmax 函数"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def forward(self, center_word_idx):
        """前向传播"""
        # 输入层到隐藏层
        self.hidden = self.W_in[center_word_idx]  # (embedding_dim,)
        
        # 隐藏层到输出层
        self.output = self.softmax(np.dot(self.hidden, self.W_out))
        
        return self.output
    
    def backward(self, center_word_idx, context_word_idx):
        """反向传播"""
        # 计算损失
        loss = -np.log(self.output[context_word_idx] + 1e-10)
        
        # 计算梯度
        grad_out = self.output.copy()
        grad_out[context_word_idx] -= 1
        
        # 更新输出权重
        grad_W_out = np.outer(self.hidden, grad_out)
        
        # 更新输入权重
        grad_hidden = np.dot(self.W_out, grad_out)
        grad_W_in = np.zeros_like(self.W_in)
        grad_W_in[center_word_idx] = grad_hidden
        
        # 参数更新
        self.W_out -= self.lr * grad_W_out
        self.W_in -= self.lr * grad_W_in
        
        return loss
    
    def train(self, corpus, window_size=2, epochs=100):
        """训练"""
        for epoch in range(epochs):
            total_loss = 0
            
            for i, center_word_idx in enumerate(corpus):
                # 获取上下文
                start = max(0, i - window_size)
                end = min(len(corpus), i + window_size + 1)
                context = [corpus[j] for j in range(start, end) if j != i]
                
                # 训练
                for context_word_idx in context:
                    self.forward(center_word_idx)
                    loss = self.backward(center_word_idx, context_word_idx)
                    total_loss += loss
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    def get_embedding(self, word_idx):
        """获取词向量"""
        return self.W_in[word_idx]

# 示例
corpus = [0, 1, 2, 3, 1, 4, 5]  # 简化语料库
vocab_size = 6
embedding_dim = 10

model = SkipGram(vocab_size, embedding_dim)
model.train(corpus, window_size=1, epochs=50)

print(f"\n词向量维度: {model.get_embedding(0).shape}")
```

---

#### 2.2 CBOW 模型

给定上下文词，预测中心词。

```
输入: 上下文词
输出: 中心词

例如："我 爱 自然语言 处理"
上下文 ["我", "自然语言"] -> 预测 "爱"
```

```python
class CBOW:
    """CBOW 模型"""
    
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_out = np.random.randn(embedding_dim, vocab_size) * 0.01
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def forward(self, context_word_indices):
        """前向传播"""
        # 平均上下文词向量
        self.hidden = np.mean(self.W_in[context_word_indices], axis=0)
        
        # 预测中心词
        self.output = self.softmax(np.dot(self.hidden, self.W_out))
        
        return self.output
    
    def train(self, corpus, window_size=2, epochs=100):
        """训练"""
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(window_size, len(corpus) - window_size):
                center_word_idx = corpus[i]
                context_indices = corpus[i-window_size:i] + corpus[i+1:i+window_size+1]
                
                # 前向传播
                output = self.forward(context_indices)
                
                # 计算损失
                loss = -np.log(output[center_word_idx] + 1e-10)
                total_loss += loss
                
                # 反向传播
                grad_out = output.copy()
                grad_out[center_word_idx] -= 1
                
                grad_W_out = np.outer(self.hidden, grad_out)
                grad_hidden = np.dot(self.W_out, grad_out)
                
                # 更新参数
                self.W_out -= self.lr * grad_W_out
                for idx in context_indices:
                    self.W_in[idx] -= self.lr * grad_hidden / len(context_indices)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
```

---

### 3. GloVe

GloVe（Global Vectors）结合全局统计信息和局部上下文。

```python
import numpy as np
from collections import defaultdict

class GloVe:
    """简化版 GloVe"""
    
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.05, x_max=100, alpha=0.75):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.x_max = x_max
        self.alpha = alpha
        
        # 词向量
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_context = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # 偏置
        self.b = np.zeros(vocab_size)
        self.b_context = np.zeros(vocab_size)
    
    def weighting_function(self, x):
        """权重函数"""
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        return 1.0
    
    def compute_cooccurrence(self, corpus, window_size=2):
        """计算共现矩阵"""
        cooccurrence = defaultdict(int)
        
        for i, word in enumerate(corpus):
            start = max(0, i - window_size)
            end = min(len(corpus), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    distance = abs(i - j)
                    cooccurrence[(word, corpus[j])] += 1.0 / distance
        
        return cooccurrence
    
    def train(self, corpus, window_size=2, epochs=100):
        """训练"""
        cooccurrence = self.compute_cooccurrence(corpus, window_size)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for (i, j), x_ij in cooccurrence.items():
                # 前向传播
                dot_product = np.dot(self.W[i], self.W_context[j])
                prediction = dot_product + self.b[i] + self.b_context[j]
                
                # 计算损失
                diff = prediction - np.log(x_ij + 1)
                weight = self.weighting_function(x_ij)
                loss = weight * diff ** 2
                total_loss += loss
                
                # 计算梯度
                grad = weight * diff
                
                # 更新参数
                grad_W_i = grad * self.W_context[j]
                grad_W_context_j = grad * self.W[i]
                
                self.W[i] -= self.lr * grad_W_i
                self.W_context[j] -= self.lr * grad_W_context_j
                self.b[i] -= self.lr * grad
                self.b_context[j] -= self.lr * grad
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    def get_embedding(self, word_idx):
        """获取词向量（主向量 + 上下文向量）"""
        return self.W[word_idx] + self.W_context[word_idx]
```

---

### 4. 词向量应用

#### 4.1 词语相似度

```python
def cosine_similarity(v1, v2):
    """余弦相似度"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 示例（假设已训练好的词向量）
word_vectors = {
    "国王": np.array([0.8, 0.2, 0.6]),
    "女王": np.array([0.7, 0.3, 0.5]),
    "男人": np.array([0.9, 0.1, 0.3]),
    "女人": np.array([0.6, 0.4, 0.2])
}

# 计算相似度
print("词语相似度:")
for w1 in word_vectors:
    for w2 in word_vectors:
        if w1 != w2:
            sim = cosine_similarity(word_vectors[w1], word_vectors[w2])
            print(f"  {w1} vs {w2}: {sim:.4f}")
```

#### 4.2 类比推理

```python
def analogy(word_a, word_b, word_c, word_vectors):
    """
    类比推理: A 之于 B 相当于 C 之于 ?
    例如: "国王" - "男人" + "女人" = "女王"
    """
    vec_a = word_vectors[word_a]
    vec_b = word_vectors[word_b]
    vec_c = word_vectors[word_c]
    
    # 计算: B - A + C
    result_vec = vec_b - vec_a + vec_c
    
    # 找最相似的词
    best_word = None
    best_sim = -1
    
    for word, vec in word_vectors.items():
        if word not in [word_a, word_b, word_c]:
            sim = cosine_similarity(result_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best_word = word
    
    return best_word

# 示例
result = analogy("男人", "国王", "女人", word_vectors)
print(f"\n类比推理: 男人之于国王，相当于女人之于 {result}")
```

---

### 5. 使用预训练词向量

#### 5.1 加载 GloVe 词向量

```python
def load_glove_vectors(glove_file):
    """加载 GloVe 预训练词向量"""
    word_vectors = {}
    
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            word_vectors[word] = vector
    
    return word_vectors

# 示例（需要下载 GloVe 文件）
# word_vectors = load_glove_vectors('glove.6B.100d.txt')
# print(f"词汇表大小: {len(word_vectors)}")
```

#### 5.2 可视化词向量

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_word_vectors(word_vectors, words):
    """可视化词向量（降维到2D）"""
    # 提取向量
    vectors = np.array([word_vectors[word] for word in words])
    
    # PCA 降维
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    for i, word in enumerate(words):
        x, y = vectors_2d[i]
        plt.scatter(x, y, marker='o', color='blue', s=100)
        plt.text(x + 0.01, y + 0.01, word, fontsize=12)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('词向量可视化')
    plt.grid(True, alpha=0.3)
    plt.show()

# 示例
words = ['国王', '女王', '男人', '女人', '王子', '公主']
# visualize_word_vectors(word_vectors, words)
```

---

## 💻 完整代码示例

### 示例：训练中文词向量

```python
"""
完整示例：从中文语料训练词向量
"""
import numpy as np
from collections import Counter, defaultdict
import jieba

class Word2VecTrainer:
    """Word2Vec 训练器"""
    
    def __init__(self, embedding_dim=100, window_size=5, min_count=5):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
    
    def build_vocab(self, sentences):
        """构建词汇表"""
        word_counts = Counter()
        
        for sentence in sentences:
            words = jieba.lcut(sentence)
            word_counts.update(words)
        
        # 过滤低频词
        self.word2idx = {}
        self.idx2word = {}
        
        for word, count in word_counts.items():
            if count >= self.min_count:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"词汇表大小: {self.vocab_size}")
    
    def generate_training_data(self, sentences):
        """生成训练数据"""
        training_data = []
        
        for sentence in sentences:
            words = jieba.lcut(sentence)
            word_indices = [self.word2idx[w] for w in words if w in self.word2idx]
            
            for i, center_idx in enumerate(word_indices):
                start = max(0, i - self.window_size)
                end = min(len(word_indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        training_data.append((center_idx, word_indices[j]))
        
        return training_data
    
    def train(self, sentences, epochs=5, learning_rate=0.025):
        """训练 Skip-gram 模型"""
        self.build_vocab(sentences)
        training_data = self.generate_training_data(sentences)
        
        # 初始化权重
        W_in = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
        W_out = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01
        
        print(f"训练数据量: {len(training_data)}")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for center_idx, context_idx in training_data:
                # 前向传播
                hidden = W_in[center_idx]
                output = np.exp(np.dot(hidden, W_out))
                output = output / output.sum()
                
                # 计算损失
                loss = -np.log(output[context_idx] + 1e-10)
                total_loss += loss
                
                # 反向传播
                grad_out = output.copy()
                grad_out[context_idx] -= 1
                
                grad_W_out = np.outer(hidden, grad_out)
                grad_hidden = np.dot(W_out, grad_out)
                
                # 更新参数
                W_out -= learning_rate * grad_W_out
                W_in[center_idx] -= learning_rate * grad_hidden
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        
        self.word_vectors = W_in
    
    def get_vector(self, word):
        """获取词向量"""
        if word in self.word2idx:
            return self.word_vectors[self.word2idx[word]]
        return None
    
    def most_similar(self, word, topn=10):
        """找最相似的词"""
        if word not in self.word2idx:
            return []
        
        word_vec = self.get_vector(word)
        similarities = []
        
        for other_word in self.word2idx:
            if other_word != word:
                other_vec = self.get_vector(other_word)
                sim = np.dot(word_vec, other_vec) / (
                    np.linalg.norm(word_vec) * np.linalg.norm(other_vec)
                )
                similarities.append((other_word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

# 示例使用
sentences = [
    "自然语言处理是人工智能的重要分支",
    "深度学习改变了自然语言处理的方式",
    "词向量是自然语言处理的基础技术",
    "机器学习是人工智能的核心技术",
    "深度学习是机器学习的一种方法"
]

trainer = Word2VecTrainer(embedding_dim=50, min_count=1)
trainer.train(sentences, epochs=50)

print("\n相似词:")
similar_words = trainer.most_similar("自然语言", topn=5)
for word, sim in similar_words:
    print(f"  {word}: {sim:.4f}")
```

---

## 🎯 实践练习

### 练习 1：实现 TF-IDF

**任务**：从零实现 TF-IDF，并在文档检索中应用。

### 练习 2：词向量类比

**任务**：使用预训练词向量进行类比推理实验。

---

## 📝 本章小结

### 核心要点

1. **文本表示**：One-Hot、词袋、TF-IDF
2. **Word2Vec**：Skip-gram 和 CBOW
3. **GloVe**：全局统计 + 局部上下文
4. **应用**：相似度计算、类比推理

---

<div align="center">

[⬅️ 上一章](../chapter04-neural-networks/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter06-rnn/README.md)

</div>
