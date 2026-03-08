"""
第3章代码示例 01：线性回归从零实现
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# 线性回归类
# ================================

class LinearRegression:
    """从零实现线性回归"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化
        
        参数:
            learning_rate: 学习率
            n_iterations: 迭代次数
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        """
        训练模型（梯度下降）
        
        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标值 (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.n_iterations):
            # 前向传播：计算预测值
            y_pred = np.dot(X, self.weights) + self.bias
            
            # 计算损失（MSE）
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # 计算梯度
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # 每100轮打印一次
            if i % 100 == 0:
                print(f"第 {i} 轮，损失: {loss:.4f}")
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 特征矩阵
        
        返回:
            预测值
        """
        return np.dot(X, self.weights) + self.bias

# ================================
# 生成数据
# ================================

np.random.seed(42)
X = np.random.randn(100, 1) * 2
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.5

print(f"数据形状: X={X.shape}, y={y.shape}")

# ================================
# 训练模型
# ================================

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# ================================
# 预测
# ================================

X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# ================================
# 可视化
# ================================

plt.figure(figsize=(12, 5))

# 散点图和回归线
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.5, label='训练数据', color='blue')
plt.plot(X_test, y_pred, 'r-', linewidth=2, label='预测线')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('线性回归拟合', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(model.loss_history, linewidth=2)
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('MSE 损失', fontsize=12)
plt.title('损失下降曲线', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression.png', dpi=300, bbox_inches='tight')
print("\n图形已保存为 linear_regression.png")
plt.show()

# ================================
# 输出结果
# ================================

print(f"\n学习到的参数:")
print(f"权重 w = {model.weights[0]:.4f} (真实值: 3.0)")
print(f"偏置 b = {model.bias:.4f} (真实值: 2.0)")
