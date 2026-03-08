"""
第3章代码示例 02：K-近邻算法
"""
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# KNN 分类器
# ================================

class KNNClassifier:
    """K-近邻分类器"""
    
    def __init__(self, k=3):
        """
        初始化
        
        参数:
            k: 邻居数量
        """
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        训练（存储数据）
        
        参数:
            X: 训练特征
            y: 训练标签
        """
        self.X_train = X
        self.y_train = y
        print(f"训练数据: {X.shape[0]} 个样本")
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 测试特征
        
        返回:
            预测标签
        """
        predictions = []
        
        for i, x in enumerate(X):
            # 计算 Euclidean 距离
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # 找到最近的 k 个邻居
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            k_distances = distances[k_indices]
            
            # 多数投票
            most_common = Counter(k_labels).most_common(1)
            predictions.append(most_common[0][0])
        
        return np.array(predictions)
    
    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# ================================
# 生成数据
# ================================

np.random.seed(42)

# 两类数据
X_class0 = np.random.randn(50, 2) + np.array([2, 2])
X_class1 = np.random.randn(50, 2) + np.array([-2, -2])
X = np.vstack([X_class0, X_class1])
y = np.array([0] * 50 + [1] * 50)

print(f"数据形状: X={X.shape}, y={y.shape}")
print(f"类别分布: 类别0={np.sum(y==0)}, 类别1={np.sum(y==1)}")

# ================================
# 训练模型
# ================================

knn = KNNClassifier(k=5)
knn.fit(X, y)

# ================================
# 创建网格可视化
# ================================

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# ================================
# 可视化
# ================================

plt.figure(figsize=(10, 8))

# 决策边界
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# 数据点
plt.scatter(X_class0[:, 0], X_class0[:, 1], 
           c='blue', label='类别 0', edgecolors='k', s=50)
plt.scatter(X_class1[:, 0], X_class1[:, 1], 
           c='red', label='类别 1', edgecolors='k', s=50)

plt.xlabel('特征 1', fontsize=12)
plt.ylabel('特征 2', fontsize=12)
plt.title('KNN 分类决策边界 (k=5)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('knn_decision_boundary.png', dpi=300, bbox_inches='tight')
print("\n图形已保存为 knn_decision_boundary.png")
plt.show()

# ================================
# 评估
# ================================

accuracy = knn.score(X, y)
print(f"\n训练集准确率: {accuracy:.4f}")
