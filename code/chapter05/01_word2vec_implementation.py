"""
第5章代码示例：词向量训练与可视化
"""
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# Word2Vec 简化实现
# ================================

class SimpleWord2Vec:
    """简化的 Word2Vec 实现"""
    
    def __init__(self, embedding_dim=10, window_size=2, learning_rate=0.01):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.lr = learning_rate
        self.word2idx = {}
        self.idx2word = {}
        self.W_in = None
        self.W_out = None
    
    def build_vocab(self, corpus):
        """构建词汇表"""
        words = [word for sentence in corpus for word in sentence.split()]
        vocab = list(set(words))
        
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        
        vocab_size = len(vocab)
        self.W_in = np.random.randn(vocab_size, self.embedding_dim) * 0.01
        self.W_out = np.random.randn(self.embedding_dim, vocab_size) * 0.01
        
        print(f"词汇表大小: {vocab_size}")
        return vocab_size
    
    def softmax(self, x):
        """Softmax 函数"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def train(self, corpus, epochs=100):
        """训练 Skip-gram 模型"""
        vocab_size = self.build_vocab(corpus)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for sentence in corpus:
                words = sentence.split()
                word_indices = [self.word2idx[w] for w in words]
                
                for i, center_idx in enumerate(word_indices):
                    # 获取上下文窗口
                    start = max(0, i - self.window_size)
                    end = min(len(word_indices), i + self.window_size + 1)
                    
                    for j in range(start, end):
                        if i != j:
                            context_idx = word_indices[j]
                            
                            # 前向传播
                            hidden = self.W_in[center_idx]
                            output = self.softmax(np.dot(hidden, self.W_out))
                            
                            # 计算损失
                            loss = -np.log(output[context_idx] + 1e-10)
                            total_loss += loss
                            
                            # 反向传播
                            grad_out = output.copy()
                            grad_out[context_idx] -= 1
                            
                            grad_W_out = np.outer(hidden, grad_out)
                            grad_hidden = np.dot(self.W_out, grad_out)
                            
                            # 更新参数
                            self.W_out -= self.lr * grad_W_out
                            self.W_in[center_idx] -= self.lr * grad_hidden
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    def get_vector(self, word):
        """获取词向量"""
        if word in self.word2idx:
            return self.W_in[self.word2idx[word]]
        return None
    
    def similarity(self, word1, word2):
        """计算词相似度"""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        
        if vec1 is not None and vec2 is not None:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return 0.0

# ================================
# 训练示例
# ================================

corpus = [
    "我 爱 自然语言 处理",
    "自然语言 处理 很 有趣",
    "我 爱 学习 深度学习",
    "深度学习 是 机器学习 的 分支"
]

print("训练 Word2Vec...")
model = SimpleWord2Vec(embedding_dim=8, window_size=1, learning_rate=0.1)
model.train(corpus, epochs=100)

print("\n词相似度:")
print(f"'我' vs '学习': {model.similarity('我', '学习'):.4f}")
print(f"'自然语言' vs '处理': {model.similarity('自然语言', '处理'):.4f}")

# ================================
# 可视化词向量
# ================================

def visualize_embeddings(model, words):
    """可视化词向量"""
    # 提取向量
    vectors = []
    labels = []
    
    for word in words:
        vec = model.get_vector(word)
        if vec is not None:
            vectors.append(vec)
            labels.append(word)
    
    if len(vectors) < 2:
        print("词汇不足，无法可视化")
        return
    
    vectors = np.array(vectors)
    
    # PCA 降维
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    for i, label in enumerate(labels):
        x, y = vectors_2d[i]
        plt.scatter(x, y, marker='o', s=100, edgecolors='k')
        plt.text(x + 0.01, y + 0.01, label, fontsize=12)
    
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.title('词向量可视化', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('word_embeddings_visualization.png', dpi=300, bbox_inches='tight')
    print("\n可视化已保存为 word_embeddings_visualization.png")
    plt.show()

# 可视化
all_words = list(model.word2idx.keys())
visualize_embeddings(model, all_words)
