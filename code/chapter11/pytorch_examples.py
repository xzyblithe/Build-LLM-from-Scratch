"""
第11章：PyTorch 深度学习框架示例
包含张量操作、自动求导、神经网络构建
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ========== 1. 张量操作 ==========

def tensor_basics():
    """张量基础操作"""
    print("1. 张量基础操作")
    print("-" * 40)
    
    # 创建张量
    x = torch.tensor([1, 2, 3, 4, 5])
    print(f"从列表创建: {x}")
    
    # 特殊张量
    zeros = torch.zeros(3, 4)
    ones = torch.ones(2, 3)
    random = torch.randn(3, 3)
    
    print(f"全零张量形状: {zeros.shape}")
    print(f"随机张量形状: {random.shape}")
    
    # 张量操作
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    
    print(f"\n加法: {(a + b).shape}")
    print(f"矩阵乘法: {torch.matmul(a, b.T).shape}")
    
    # GPU 支持
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    x_gpu = x.to(device)
    print(f"张量设备: {x_gpu.device}")


# ========== 2. 自动求导 ==========

def autograd_example():
    """自动求导示例"""
    print("\n2. 自动求导")
    print("-" * 40)
    
    # 创建需要梯度的张量
    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)
    
    # 构建计算图
    z = x * y + x ** 2
    
    print(f"z = x*y + x^2 = {z.item()}")
    
    # 反向传播
    z.backward()
    
    print(f"∂z/∂x = y + 2x = {x.grad.item()}")
    print(f"∂z/∂y = x = {y.grad.item()}")


# ========== 3. 神经网络 ==========

class SimpleMLP(nn.Module):
    """简单的多层感知机"""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TextClassifier(nn.Module):
    """文本分类模型"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids):
        # 嵌入
        embedded = self.embedding(input_ids)
        
        # LSTM
        output, (hidden, _) = self.lstm(embedded)
        
        # 取最后隐藏状态
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # 分类
        logits = self.classifier(hidden)
        
        return logits


def neural_network_example():
    """神经网络示例"""
    print("\n3. 神经网络")
    print("-" * 40)
    
    # 创建模型
    model = SimpleMLP(784, [256, 128], 10)
    
    # 打印模型结构
    print(model)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")
    
    # 测试前向传播
    x = torch.randn(32, 784)
    output = model(x)
    print(f"输入形状: {x.shape}, 输出形状: {output.shape}")


# ========== 4. 训练循环 ==========

class SimpleDataset(Dataset):
    """简单数据集"""
    
    def __init__(self, n_samples=1000, input_dim=10):
        self.x = torch.randn(n_samples, input_dim)
        self.y = torch.randint(0, 2, (n_samples,))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def training_loop():
    """完整训练循环"""
    print("\n4. 训练循环")
    print("-" * 40)
    
    # 数据
    dataset = SimpleDataset(1000, 10)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 模型
    model = SimpleMLP(10, [32, 16], 2)
    
    # 损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    for epoch in range(3):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")


# ========== 5. 模型保存加载 ==========

def save_load_example():
    """模型保存加载示例"""
    print("\n5. 模型保存加载")
    print("-" * 40)
    
    model = SimpleMLP(10, [32], 2)
    
    # 保存参数
    torch.save(model.state_dict(), 'model_weights.pt')
    print("模型参数已保存到 model_weights.pt")
    
    # 加载参数
    new_model = SimpleMLP(10, [32], 2)
    new_model.load_state_dict(torch.load('model_weights.pt'))
    print("模型参数已加载")
    
    # 保存完整模型
    torch.save(model, 'model_complete.pt')
    print("完整模型已保存")


# ========== 主函数 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch 深度学习框架示例")
    print("=" * 60)
    
    tensor_basics()
    autograd_example()
    neural_network_example()
    training_loop()
    save_load_example()
    
    print("\n✅ PyTorch 示例完成!")
