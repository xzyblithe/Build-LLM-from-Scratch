# 第11章：PyTorch 深度学习框架

<div align="center">

[⬅️ 上一章](../chapter10-llm-principles/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter12-hugging-face/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解 PyTorch 的核心概念与张量操作
- ✅ 掌握自动求导机制（Autograd）
- ✅ 构建和训练神经网络模型
- ✅ 使用 DataLoader 进行数据处理
- ✅ 理解 GPU 加速与模型部署

---

## 🎯 本章内容

### 1. PyTorch 简介

#### 1.1 什么是 PyTorch？

PyTorch 是 Facebook（Meta）开源的深度学习框架，以其动态计算图和易用性著称。

```
PyTorch 核心特性：

1. 动态计算图（Dynamic Computation Graph）
   - 边运行边构建计算图
   - 便于调试和灵活设计

2. 张量计算（Tensor Computation）
   - 类似 NumPy，但支持 GPU 加速
   - 丰富的数学运算

3. 自动求导（Autograd）
   - 自动计算梯度
   - 简化反向传播实现

4. 丰富的生态系统
   - torchvision, torchtext, torchaudio
   - Hugging Face, PyTorch Lightning
```

#### 1.2 PyTorch vs TensorFlow

| 特性 | PyTorch | TensorFlow |
|------|---------|------------|
| 计算图 | 动态（Eager） | 静态 + 动态 |
| 调试 | Python 原生 | 较复杂 |
| 学习曲线 | 平缓 | 较陡 |
| 社区 | 研究主导 | 工业主导 |
| 部署 | TorchScript | TensorFlow Serving |

---

### 2. 张量（Tensor）

#### 2.1 创建张量

```python
import torch
import numpy as np

# 从列表创建
x = torch.tensor([1, 2, 3, 4, 5])
print(f"张量: {x}")
print(f"形状: {x.shape}")
print(f"数据类型: {x.dtype}")

# 创建特定形状的张量
zeros = torch.zeros(3, 4)      # 全零
ones = torch.ones(2, 3)        # 全一
random = torch.randn(3, 3)     # 标准正态分布
arange = torch.arange(0, 10, 2)  # 等差数列

print(f"\n全零张量:\n{zeros}")
print(f"\n随机张量:\n{random}")

# 从 NumPy 转换
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)
print(f"\n从 NumPy 转换: {tensor}")

# 从张量转 NumPy
back_to_numpy = tensor.numpy()
print(f"转回 NumPy: {back_to_numpy}")
```

#### 2.2 张量操作

```python
import torch

# 创建张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([[7, 8, 9], [10, 11, 12]])

print(f"x:\n{x}")
print(f"x.shape: {x.shape}")

# 索引和切片
print(f"\n第一行: {x[0]}")
print(f"第一列: {x[:, 0]}")
print(f"右下角元素: {x[1, 2]}")

# 形状操作
print(f"\n展平: {x.flatten()}")
print(f"重塑为 (3, 2):\n{x.reshape(3, 2)}")
print(f"转置:\n{x.T}")

# 数学运算
print(f"\n加法:\n{x + y}")
print(f"逐元素乘法:\n{x * y}")
print(f"矩阵乘法:\n{torch.matmul(x, y.T)}")

# 统计操作
print(f"\n求和: {x.sum()}")
print(f"均值: {x.float().mean()}")
print(f"最大值: {x.max()}")
print(f"每列最大值: {x.max(dim=0)}")
```

#### 2.3 GPU 加速

```python
import torch

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建张量并移动到 GPU
x = torch.randn(1000, 1000)
x_gpu = x.to(device)

# GPU 上的运算
y_gpu = torch.matmul(x_gpu, x_gpu.T)

# 移回 CPU
y_cpu = y_gpu.cpu()

# 性能对比
import time

# CPU 矩阵乘法
x = torch.randn(5000, 5000)
start = time.time()
z = torch.matmul(x, x.T)
cpu_time = time.time() - start
print(f"\nCPU 时间: {cpu_time:.4f}s")

# GPU 矩阵乘法（如果可用）
if torch.cuda.is_available():
    x_gpu = x.cuda()
    torch.cuda.synchronize()  # 同步
    start = time.time()
    z_gpu = torch.matmul(x_gpu, x_gpu.T)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU 时间: {gpu_time:.4f}s")
    print(f"加速比: {cpu_time / gpu_time:.2f}x")
```

---

### 3. 自动求导（Autograd）

#### 3.1 基本原理

```python
import torch

# 创建需要梯度的张量
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# 构建计算图
z = x * y + x ** 2

print(f"z = x*y + x^2 = {z.item()}")

# 反向传播
z.backward()

print(f"\n∂z/∂x = y + 2x = {x.grad.item()}")  # 应该是 3 + 4 = 7
print(f"∂z/∂y = x = {y.grad.item()}")  # 应该是 2
```

#### 3.2 计算图可视化

```
计算图示例：z = x * y + x^2

        x (requires_grad=True)
       / \
      |   \
      |    \
      v     v
    x^2    x*y
      \     /
       \   /
        \ /
         +
         |
         v
         z

反向传播时，梯度沿着边反向流动
```

#### 3.3 梯度控制

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 方式1：no_grad 上下文管理器
with torch.no_grad():
    y = x * 2
    print(f"y.requires_grad (在 no_grad 中): {y.requires_grad}")

# 方式2：detach() 分离张量
y = x * 2
y_detached = y.detach()
print(f"y.requires_grad: {y.requires_grad}")
print(f"y_detached.requires_grad: {y_detached.requires_grad}")

# 方式3：禁用梯度计算全局设置
torch.set_grad_enabled(False)
y = x * 2
print(f"全局禁用后 y.requires_grad: {y.requires_grad}")
torch.set_grad_enabled(True)

# 清零梯度
x.grad = None  # 或 x.grad.zero_()
```

---

### 4. 神经网络模块（nn.Module）

#### 4.1 构建神经网络

```python
import torch
import torch.nn as nn

# 方式1：使用 nn.Sequential
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

print(model)

# 方式2：自定义类
class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 创建模型
model = MLP(784, [256, 128], 10)
print(f"\n自定义模型:\n{model}")

# 查看参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
```

#### 4.2 常用网络层

```python
import torch.nn as nn

# 线性层
linear = nn.Linear(64, 32)

# 卷积层
conv1d = nn.Conv1d(3, 16, kernel_size=3)  # 文本
conv2d = nn.Conv2d(3, 16, kernel_size=3)  # 图像

# 循环层
rnn = nn.RNN(64, 128, batch_first=True)
lstm = nn.LSTM(64, 128, batch_first=True)
gru = nn.GRU(64, 128, batch_first=True)

# Transformer
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# 归一化层
bn = nn.BatchNorm1d(64)
ln = nn.LayerNorm(64)

# 正则化层
dropout = nn.Dropout(0.5)

# 激活函数
relu = nn.ReLU()
gelu = nn.GELU()
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=-1)

# 嵌入层
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=256)

# 注意力
attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
```

---

### 5. 损失函数与优化器

#### 5.1 常用损失函数

```python
import torch
import torch.nn as nn

# 分类任务
cross_entropy = nn.CrossEntropyLoss()
bce_loss = nn.BCELoss()
bce_with_logits = nn.BCEWithLogitsLoss()

# 示例：交叉熵损失
predictions = torch.randn(4, 10)  # 4个样本，10个类别
labels = torch.tensor([0, 2, 5, 9])  # 真实标签
loss = cross_entropy(predictions, labels)
print(f"交叉熵损失: {loss.item()}")

# 回归任务
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
smooth_l1 = nn.SmoothL1Loss()

# 示例：MSE 损失
predictions = torch.randn(4, 1)
targets = torch.randn(4, 1)
loss = mse_loss(predictions, targets)
print(f"MSE 损失: {loss.item()}")

# 序列任务
ctc_loss = nn.CTCLoss()

# 对比学习
triplet_loss = nn.TripletMarginLoss(margin=1.0)
```

#### 5.2 优化器

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 1)

# SGD
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)

# SGD with Momentum
optimizer_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer_adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW（带权重衰减）
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 学习率调度器
scheduler_step = optim.lr_scheduler.StepLR(optimizer_adam, step_size=10, gamma=0.1)
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=100)
scheduler_warmup = optim.lr_scheduler.LinearLR(
    optimizer_adam, start_factor=0.1, total_iters=10
)

# 训练循环示例
def train_step(model, optimizer, x, y):
    """单步训练"""
    # 清零梯度
    optimizer.zero_grad()
    
    # 前向传播
    predictions = model(x)
    loss = nn.MSELoss()(predictions, y)
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    optimizer.step()
    
    return loss.item()
```

---

### 6. 数据加载（DataLoader）

#### 6.1 Dataset 类

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """自定义数据集"""
    
    def __init__(self, data, labels, transform=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


# 示例数据
import numpy as np
data = np.random.randn(1000, 10)
labels = np.random.randint(0, 2, 1000)

# 创建数据集
dataset = CustomDataset(data, labels)
print(f"数据集大小: {len(dataset)}")
print(f"第一个样本: {dataset[0]}")
```

#### 6.2 DataLoader

```python
from torch.utils.data import DataLoader, random_split

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,      # 打乱数据
    num_workers=4,     # 多进程加载
    pin_memory=True    # 固定内存（GPU 加速）
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# 遍历数据
for batch_idx, (x, y) in enumerate(train_loader):
    print(f"Batch {batch_idx}: x.shape={x.shape}, y.shape={y.shape}")
    if batch_idx >= 2:
        break
```

#### 6.3 数据增强

```python
from torchvision import transforms

# 图像数据增强
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# 文本数据增强（简单示例）
class TextAugmentation:
    def __init__(self, vocab, p_drop=0.1):
        self.vocab = vocab
        self.p_drop = p_drop
    
    def random_delete(self, tokens):
        """随机删除词"""
        return [t for t in tokens if torch.rand(1).item() > self.p_drop]
    
    def random_swap(self, tokens):
        """随机交换词"""
        if len(tokens) < 2:
            return tokens
        i, j = torch.randint(0, len(tokens), (2,)).tolist()
        tokens[i], tokens[j] = tokens[j], tokens[i]
        return tokens
```

---

### 7. 完整训练流程

#### 7.1 训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    """完整训练流程"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    
    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
        
        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
        
        # 计算指标
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 更新学习率
        scheduler.step()
        
        # 打印进度
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  保存最佳模型 (Val Acc: {val_acc:.4f})")
    
    return history


# 示例：训练一个简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.network(x)

# 创建数据
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32)

# 训练
model = SimpleModel()
history = train_model(model, train_loader, val_loader, epochs=5)
```

#### 7.2 模型保存与加载

```python
import torch

# 保存完整模型
torch.save(model, 'model_complete.pt')

# 只保存参数（推荐）
torch.save(model.state_dict(), 'model_weights.pt')

# 加载模型
model.load_state_dict(torch.load('model_weights.pt'))
model.eval()  # 设置为评估模式

# 保存检查点（包含优化器状态）
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pt')

# 加载检查点
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

### 8. 模型部署

#### 8.1 TorchScript

```python
import torch

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.linear(x)

model = MyModel()
model.eval()

# 方式1：tracing
example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input)
print(f"Traced model:\n{traced_model}")

# 方式2：scripting
scripted_model = torch.jit.script(model)
print(f"\nScripted model:\n{scripted_model}")

# 保存
traced_model.save('traced_model.pt')
scripted_model.save('scripted_model.pt')

# 加载
loaded_model = torch.jit.load('traced_model.pt')
output = loaded_model(torch.randn(1, 10))
print(f"\n输出: {output}")
```

#### 8.2 ONNX 导出

```python
import torch

model = MyModel()
model.eval()

# 导出为 ONNX
dummy_input = torch.randn(1, 10)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("ONNX 模型已导出")
```

---

## 💻 完整代码示例

### 示例：文本分类模型

```python
"""
完整的文本分类模型实现
包含数据处理、模型定义、训练和评估
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ========== 数据集 ==========
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # 简单分词
        tokens = self.texts[idx].lower().split()[:self.max_length]
        
        # 转换为索引
        indices = [self.vocab.get(t, 0) for t in tokens]  # 0 for <unk>
        
        # 填充
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))
        
        return torch.tensor(indices), torch.tensor(self.labels[idx])


# ========== 模型 ==========
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len)
        
        # 嵌入
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # LSTM
        output, (hidden, cell) = self.lstm(embedded)
        
        # 取最后一层的双向隐藏状态
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, hidden_dim*2)
        
        # 分类
        logits = self.classifier(hidden)
        
        return logits


# ========== 训练函数 ==========
def train_text_classifier():
    """训练文本分类器"""
    
    # 模拟数据
    texts = [
        "这部电影很好看 我很喜欢",
        "太糟糕了 浪费时间",
        "非常推荐 值得一看",
        "无聊至极 不推荐",
    ] * 100
    
    labels = [1, 0, 1, 0] * 100  # 1: 正面, 0: 负面
    
    # 构建词汇表
    vocab = {}
    idx = 1  # 0 保留给 <unk>
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    
    print(f"词汇表大小: {len(vocab)}")
    
    # 创建数据集
    dataset = TextClassificationDataset(texts, labels, vocab, max_length=20)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextClassifier(
        vocab_size=len(vocab) + 1,
        embed_dim=64,
        hidden_dim=32,
        num_classes=2,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # 损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, "
              f"Acc: {correct/total:.4f}")
    
    print("\n训练完成!")
    return model


# 运行训练
if __name__ == "__main__":
    model = train_text_classifier()
```

---

## 🎯 实践练习

### 练习 1：实现 ResNet 残差块

```python
# TODO: 实现 ResNet 的残差连接块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 实现...
    
    def forward(self, x):
        # 实现...
        pass
```

### 练习 2：实现自定义学习率调度器

```python
# TODO: 实现带预热的余弦退火调度器
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        # 实现...
        pass
```

---

## 📝 本章小结

### 核心要点

1. **张量**：PyTorch 的基本数据结构，支持 GPU 加速
2. **Autograd**：自动求导机制，简化梯度计算
3. **nn.Module**：构建神经网络的基础类
4. **DataLoader**：高效的数据加载和批处理
5. **训练循环**：前向传播、反向传播、参数更新的标准流程

### 关键概念

- 动态计算图（Dynamic Computation Graph）
- requires_grad 与梯度追踪
- model.train() vs model.eval()
- GPU 加速与设备转移
- TorchScript 模型部署

---

<div align="center">

[⬅️ 上一章](../chapter10-llm-principles/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter12-hugging-face/README.md)

</div>
