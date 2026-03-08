# 第1章：Python 基础与开发环境搭建

<div align="center">

[⬅️ 返回目录](../README.md) | [下一章 ➡️](../chapter02-math-foundations/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 掌握 Python 基本语法和常用数据结构
- ✅ 搭建深度学习开发环境
- ✅ 使用 NumPy 进行数值计算
- ✅ 使用 Matplotlib 进行数据可视化
- ✅ 理解 Python 在机器学习中的应用

---

## 🎯 本章内容

### 1. Python 安装与环境配置

#### 1.1 为什么选择 Python？

Python 是机器学习和深度学习最流行的编程语言，原因如下：

- **简洁易学**：语法简单，接近自然语言
- **生态丰富**：NumPy、Pandas、PyTorch 等强大库
- **社区活跃**：问题容易找到解决方案
- **应用广泛**：从数据分析到深度学习全覆盖

#### 1.2 安装 Anaconda

**推荐使用 Anaconda**，它集成了 Python 和常用的科学计算库。

**下载地址**：https://www.anaconda.com/download

**安装步骤**：

```bash
# 1. 下载 Anaconda 安装包
# 2. 运行安装程序
# 3. 选择安装路径（建议默认）
# 4. 勾选 "Add Anaconda to PATH"
```

**验证安装**：

```bash
# 打开终端，输入
python --version
# 输出：Python 3.10.x

conda --version
# 输出：conda 23.x.x
```

#### 1.3 虚拟环境管理

**为什么需要虚拟环境？**

- 隔离不同项目的依赖
- 避免版本冲突
- 方便环境迁移

**创建虚拟环境**：

```bash
# 创建名为 llm-tutorial 的环境，Python 版本 3.10
conda create -n llm-tutorial python=3.10

# 激活环境
conda activate llm-tutorial

# 退出环境
conda deactivate

# 查看所有环境
conda env list

# 删除环境
conda env remove -n llm-tutorial
```

#### 1.4 Jupyter Notebook 使用

Jupyter Notebook 是交互式编程环境，非常适合学习和实验。

**启动 Jupyter**：

```bash
# 激活虚拟环境
conda activate llm-tutorial

# 启动 Jupyter Notebook
jupyter notebook

# 或启动 Jupyter Lab（更现代的界面）
jupyter lab
```

**Jupyter 常用快捷键**：

| 快捷键 | 功能 |
|--------|------|
| `Shift + Enter` | 运行当前单元格，跳到下一个 |
| `Ctrl + Enter` | 运行当前单元格 |
| `Esc` | 进入命令模式 |
| `Enter` | 进入编辑模式 |
| `A` | 在上方插入单元格（命令模式） |
| `B` | 在下方插入单元格（命令模式） |
| `DD` | 删除单元格（命令模式） |
| `M` | 转为 Markdown 单元格 |
| `Y` | 转为 Code 单元格 |

---

### 2. Python 核心语法速成

#### 2.1 数据类型与变量

Python 有以下基本数据类型：

```python
# 整数（Integer）
age = 25
print(type(age))  # <class 'int'>

# 浮点数（Float）
height = 1.75
print(type(height))  # <class 'float'>

# 字符串（String）
name = "Alice"
print(type(name))  # <class 'str'>

# 布尔值（Boolean）
is_student = True
print(type(is_student))  # <class 'bool'>

# 空值
empty = None
print(type(empty))  # <class 'NoneType'>
```

**类型转换**：

```python
# 字符串转整数
num_str = "123"
num_int = int(num_str)
print(num_int + 1)  # 124

# 整数转字符串
num = 456
num_str = str(num)
print(num_str + " apples")  # "456 apples"

# 浮点数转整数（截断）
pi = 3.14159
pi_int = int(pi)
print(pi_int)  # 3
```

#### 2.2 列表（List）

列表是 Python 最常用的数据结构，类似数组但更灵活。

```python
# 创建列表
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]  # 可以混合类型

# 访问元素（索引从 0 开始）
print(fruits[0])   # "apple"
print(fruits[-1])  # "orange"（最后一个）
print(fruits[-2])  # "banana"（倒数第二个）

# 切片（Slicing）
print(numbers[1:4])   # [2, 3, 4]
print(numbers[:3])    # [1, 2, 3]
print(numbers[2:])    # [3, 4, 5]
print(numbers[::2])   # [1, 3, 5]（步长为2）

# 修改列表
fruits[0] = "grape"       # 修改元素
fruits.append("mango")    # 添加元素
fruits.insert(1, "pear")  # 插入元素
fruits.remove("banana")   # 删除元素
popped = fruits.pop()     # 弹出最后一个元素

# 列表操作
print(len(fruits))        # 列表长度
print("apple" in fruits)  # 检查是否存在
fruits.sort()             # 排序
fruits.reverse()          # 反转

# 列表推导式（List Comprehension）
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

evens = [x for x in range(20) if x % 2 == 0]
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

#### 2.3 字典（Dictionary）

字典是键值对（Key-Value）的数据结构。

```python
# 创建字典
person = {
    "name": "Alice",
    "age": 25,
    "city": "Beijing"
}

# 访问值
print(person["name"])       # "Alice"
print(person.get("age"))    # 25
print(person.get("job", "Unknown"))  # "Unknown"（默认值）

# 修改字典
person["age"] = 26               # 修改值
person["job"] = "Engineer"       # 添加键值对
del person["city"]               # 删除键值对

# 遍历字典
for key in person:
    print(key, person[key])

for key, value in person.items():
    print(f"{key}: {value}")

# 字典推导式
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

#### 2.4 集合（Set）

集合是无序、不重复的元素集合。

```python
# 创建集合
fruits = {"apple", "banana", "orange"}
numbers = set([1, 2, 2, 3, 3, 3])  # {1, 2, 3}（去重）

# 集合操作
fruits.add("mango")          # 添加元素
fruits.remove("banana")      # 删除元素

# 集合运算
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)   # 并集：{1, 2, 3, 4, 5, 6}
print(a & b)   # 交集：{3, 4}
print(a - b)   # 差集：{1, 2}
print(a ^ b)   # 对称差：{1, 2, 5, 6}
```

#### 2.5 函数

函数是组织代码的基本单元。

```python
# 定义函数
def greet(name):
    """问候函数"""
    return f"Hello, {name}!"

# 调用函数
message = greet("Alice")
print(message)  # "Hello, Alice!"

# 默认参数
def power(base, exponent=2):
    return base ** exponent

print(power(3))      # 9
print(power(3, 3))   # 27

# 可变参数
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4, 5))  # 15

# 关键字参数
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="Beijing")

# Lambda 表达式（匿名函数）
square = lambda x: x ** 2
print(square(5))  # 25

# 在列表排序中使用
students = [("Alice", 85), ("Bob", 92), ("Charlie", 78)]
students.sort(key=lambda x: x[1], reverse=True)
# [('Bob', 92), ('Alice', 85), ('Charlie', 78)]
```

#### 2.6 类与对象

Python 是面向对象的编程语言。

```python
# 定义类
class Student:
    """学生类"""
    
    # 构造函数
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    # 实例方法
    def introduce(self):
        return f"I'm {self.name}, {self.age} years old."
    
    # 类方法
    @classmethod
    def from_birth_year(cls, name, birth_year):
        age = 2024 - birth_year
        return cls(name, age)
    
    # 静态方法
    @staticmethod
    def is_adult(age):
        return age >= 18

# 创建对象
student1 = Student("Alice", 20)
print(student1.introduce())  # "I'm Alice, 20 years old."

# 使用类方法
student2 = Student.from_birth_year("Bob", 2000)
print(student2.age)  # 24

# 使用静态方法
print(Student.is_adult(20))  # True
```

---

### 3. NumPy 数组操作

NumPy 是 Python 科学计算的基础库，提供了高效的数组操作。

#### 3.1 NumPy 数组基础

```python
import numpy as np

# 创建数组
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])  # 二维数组

# 特殊数组
zeros = np.zeros((3, 4))        # 3x4 全零数组
ones = np.ones((2, 3))          # 2x3 全一数组
empty = np.empty((2, 2))        # 未初始化数组
arange = np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5) # [0., 0.25, 0.5, 0.75, 1.]

# 数组属性
print(arr2.shape)   # (2, 3) - 形状
print(arr2.ndim)    # 2 - 维度
print(arr2.size)    # 6 - 元素个数
print(arr2.dtype)   # int64 - 数据类型

# 改变形状
arr3 = np.arange(12)
arr3_reshaped = arr3.reshape(3, 4)  # 3x4 矩阵
arr3_flattened = arr3_reshaped.flatten()  # 展平
```

#### 3.2 数组运算

```python
# 创建数组
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# 基本运算（逐元素）
print(a + b)   # [6, 8, 10, 12]
print(a - b)   # [-4, -4, -4, -4]
print(a * b)   # [5, 12, 21, 32]
print(a / b)   # [0.2, 0.333..., 0.428..., 0.5]
print(a ** 2)  # [1, 4, 9, 16]

# 矩阵运算
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))      # 矩阵乘法
print(A @ B)             # 矩阵乘法（Python 3.5+）
print(A.T)               # 转置
print(np.linalg.inv(A))  # 逆矩阵

# 统计运算
arr = np.random.randn(1000)  # 1000 个随机数
print(np.mean(arr))     # 平均值
print(np.std(arr))      # 标准差
print(np.min(arr))      # 最小值
print(np.max(arr))      # 最大值
print(np.sum(arr))      # 求和
```

#### 3.3 数组索引与切片

```python
# 一维数组索引
arr = np.arange(10)
print(arr[0])     # 0
print(arr[-1])    # 9
print(arr[2:5])   # [2, 3, 4]

# 二维数组索引
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[0, 0])    # 1
print(arr2d[1, :])    # [4, 5, 6]
print(arr2d[:, 1])    # [2, 5, 8]
print(arr2d[0:2, 1:3])  # [[2, 3], [5, 6]]

# 布尔索引
arr = np.array([1, 2, 3, 4, 5, 6])
mask = arr > 3
print(arr[mask])  # [4, 5, 6]
print(arr[arr % 2 == 0])  # [2, 4, 6]
```

---

### 4. Matplotlib 数据可视化

Matplotlib 是 Python 最常用的绘图库。

#### 4.1 基础绘图

```python
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 负号显示

# 创建数据
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制曲线
plt.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='red', linewidth=2, linestyle='--')

# 添加标题和标签
plt.title('正弦和余弦函数', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)

# 添加图例
plt.legend(fontsize=12)

# 添加网格
plt.grid(True, alpha=0.3)

# 显示图形
plt.show()
```

#### 4.2 子图绘制

```python
# 创建 2x2 的子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 第一个子图：折线图
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('正弦函数')

# 第二个子图：散点图
x = np.random.randn(100)
y = np.random.randn(100)
axes[0, 1].scatter(x, y, alpha=0.5)
axes[0, 1].set_title('散点图')

# 第三个子图：柱状图
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 2, 5]
axes[1, 0].bar(categories, values)
axes[1, 0].set_title('柱状图')

# 第四个子图：直方图
data = np.random.randn(1000)
axes[1, 1].hist(data, bins=30, alpha=0.7)
axes[1, 1].set_title('直方图')

plt.tight_layout()
plt.show()
```

---

## 💻 完整代码示例

### 示例 1：Hello Python

```python
"""
示例 1：Python 基础语法演示
"""

# ================================
# 1. 基本数据类型
# ================================

# 整数
age = 25
print(f"年龄: {age}, 类型: {type(age)}")

# 浮点数
height = 1.75
print(f"身高: {height}m, 类型: {type(height)}")

# 字符串
name = "Alice"
print(f"姓名: {name}, 类型: {type(name)}")

# 布尔值
is_student = True
print(f"是否学生: {is_student}, 类型: {type(is_student)}")

print("\n" + "="*50 + "\n")

# ================================
# 2. 列表操作
# ================================

# 创建列表
fruits = ["apple", "banana", "orange"]
print(f"初始列表: {fruits}")

# 添加元素
fruits.append("mango")
print(f"添加后: {fruits}")

# 切片
print(f"前两个: {fruits[:2]}")
print(f"后两个: {fruits[-2:]}")

# 列表推导式
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
print(f"平方: {squares}")

print("\n" + "="*50 + "\n")

# ================================
# 3. 字典操作
# ================================

# 创建字典
person = {
    "name": "Alice",
    "age": 25,
    "city": "Beijing"
}

print(f"个人信息: {person}")

# 遍历字典
print("\n遍历字典:")
for key, value in person.items():
    print(f"  {key}: {value}")

print("\n" + "="*50 + "\n")

# ================================
# 4. 函数定义
# ================================

def fibonacci(n):
    """
    计算斐波那契数列的前 n 项
    
    参数:
        n: 项数
    
    返回:
        斐波那契数列列表
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib

# 调用函数
fib_sequence = fibonacci(10)
print(f"斐波那契数列前10项: {fib_sequence}")

print("\n" + "="*50 + "\n")

# ================================
# 5. 类定义
# ================================

class Calculator:
    """简单的计算器类"""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """加法"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """乘法"""
        result = a * b
        self.history.append(f"{a} × {b} = {result}")
        return result
    
    def show_history(self):
        """显示历史记录"""
        print("计算历史:")
        for record in self.history:
            print(f"  {record}")

# 使用计算器
calc = Calculator()
print(f"3 + 5 = {calc.add(3, 5)}")
print(f"4 × 6 = {calc.multiply(4, 6)}")
calc.show_history()
```

**运行结果**：

```
年龄: 25, 类型: <class 'int'>
身高: 1.75m, 类型: <class 'float'>
姓名: Alice, 类型: <class 'str'>
是否学生: True, 类型: <class 'bool'>

==================================================

初始列表: ['apple', 'banana', 'orange']
添加后: ['apple', 'banana', 'orange', 'mango']
前两个: ['apple', 'banana']
后两个: ['orange', 'mango']
平方: [1, 4, 9, 16, 25]

==================================================

个人信息: {'name': 'Alice', 'age': 25, 'city': 'Beijing'}

遍历字典:
  name: Alice
  age: 25
  city: Beijing

==================================================

斐波那契数列前10项: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

==================================================

3 + 5 = 8
4 × 6 = 24
计算历史:
  3 + 5 = 8
  4 × 6 = 24
```

---

### 示例 2：NumPy 基础

```python
"""
示例 2：NumPy 数组操作演示
"""
import numpy as np

# ================================
# 1. 数组创建
# ================================

print("=== 数组创建 ===")

# 从列表创建
arr1 = np.array([1, 2, 3, 4, 5])
print(f"一维数组: {arr1}")

# 二维数组
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"二维数组:\n{arr2}")

# 特殊数组
zeros = np.zeros((3, 3))
print(f"全零矩阵:\n{zeros}")

ones = np.ones((2, 4))
print(f"全一矩阵:\n{ones}")

# 随机数组
random_arr = np.random.randn(3, 3)
print(f"随机矩阵:\n{random_arr}")

print("\n" + "="*50 + "\n")

# ================================
# 2. 数组运算
# ================================

print("=== 数组运算 ===")

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"a ** 2 = {a ** 2}")

# 矩阵乘法
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"\n矩阵 A:\n{A}")
print(f"矩阵 B:\n{B}")
print(f"矩阵乘法 A @ B:\n{A @ B}")

print("\n" + "="*50 + "\n")

# ================================
# 3. 数组统计
# ================================

print("=== 数组统计 ===")

data = np.random.randn(1000)

print(f"数据个数: {len(data)}")
print(f"平均值: {np.mean(data):.4f}")
print(f"标准差: {np.std(data):.4f}")
print(f"最小值: {np.min(data):.4f}")
print(f"最大值: {np.max(data):.4f}")
print(f"中位数: {np.median(data):.4f}")

print("\n" + "="*50 + "\n")

# ================================
# 4. 数组索引与切片
# ================================

print("=== 数组索引与切片 ===")

arr = np.arange(12).reshape(3, 4)
print(f"原数组:\n{arr}")

print(f"\n第一行: {arr[0, :]}")
print(f"第一列: {arr[:, 0]}")
print(f"中心 2x2:\n{arr[1:3, 1:3]}")

# 布尔索引
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"\n原数组: {data}")
print(f"大于 5 的元素: {data[data > 5]}")
print(f"偶数: {data[data % 2 == 0]}")
```

---

### 示例 3：Matplotlib 可视化

```python
"""
示例 3：Matplotlib 数据可视化演示
"""
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# 创建数据
# ================================

x = np.linspace(0, 2*np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

# ================================
# 绘制图形
# ================================

plt.figure(figsize=(15, 10))

# 子图 1: 折线图
plt.subplot(2, 2, 1)
plt.plot(x, y_sin, label='sin(x)', color='blue', linewidth=2)
plt.plot(x, y_cos, label='cos(x)', color='red', linewidth=2, linestyle='--')
plt.title('三角函数')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图 2: 散点图
plt.subplot(2, 2, 2)
x_rand = np.random.randn(100)
y_rand = np.random.randn(100)
colors = np.random.rand(100)
plt.scatter(x_rand, y_rand, c=colors, alpha=0.6, cmap='viridis')
plt.title('随机散点图')
plt.colorbar(label='颜色值')

# 子图 3: 柱状图
plt.subplot(2, 2, 3)
categories = ['A', 'B', 'C', 'D', 'E']
values = [3, 7, 2, 5, 8]
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
plt.bar(categories, values, color=colors_bar)
plt.title('柱状图')
plt.xlabel('类别')
plt.ylabel('数值')

# 子图 4: 直方图
plt.subplot(2, 2, 4)
data = np.random.randn(1000)
plt.hist(data, bins=30, color='#6C5CE7', alpha=0.7, edgecolor='black')
plt.title('正态分布直方图')
plt.xlabel('值')
plt.ylabel('频数')

plt.tight_layout()
plt.savefig('matplotlib_demo.png', dpi=300, bbox_inches='tight')
print("图形已保存为 matplotlib_demo.png")
plt.show()
```

---

## 🎯 实践练习

### 练习 1：数据统计小程序

**任务**：编写一个程序，统计一个数字列表的基本统计信息。

**要求**：
- 计算平均值、最大值、最小值
- 计算中位数和标准差
- 找出所有大于平均值的数字

**参考代码**：

```python
import numpy as np

def analyze_numbers(numbers):
    """
    分析数字列表的统计信息
    
    参数:
        numbers: 数字列表
    
    返回:
        统计信息字典
    """
    arr = np.array(numbers)
    
    stats = {
        'count': len(arr),
        'mean': np.mean(arr),
        'median': np.median(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'above_mean': arr[arr > np.mean(arr)].tolist()
    }
    
    return stats

# 测试
numbers = [23, 45, 12, 67, 34, 89, 56, 78, 90, 11]
stats = analyze_numbers(numbers)

print("=== 数据统计分析 ===")
for key, value in stats.items():
    print(f"{key}: {value}")
```

---

### 练习 2：文本处理工具

**任务**：编写一个函数，统计英文文本中单词的出现频率。

**要求**：
- 将文本转换为小写
- 去除标点符号
- 统计每个单词的出现次数
- 返回前 N 个高频单词

**参考代码**：

```python
import re
from collections import Counter

def word_frequency(text, top_n=10):
    """
    统计文本中单词的词频
    
    参数:
        text: 英文文本
        top_n: 返回前N个高频词
    
    返回:
        词频列表
    """
    # 转小写
    text = text.lower()
    
    # 去除标点符号，只保留字母和空格
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 分词
    words = text.split()
    
    # 统计词频
    word_counts = Counter(words)
    
    # 返回前 top_n 个
    return word_counts.most_common(top_n)

# 测试
text = """
Python is a high-level programming language. 
Python is widely used in machine learning and data science.
Learning Python is fun and useful.
"""

top_words = word_frequency(text, top_n=5)

print("=== 词频统计 ===")
for word, count in top_words:
    print(f"{word}: {count}")
```

---

## 📝 本章小结

### 核心要点

1. **Python 环境**：使用 Anaconda 管理环境和依赖
2. **基础语法**：掌握列表、字典、集合等数据结构
3. **NumPy**：高效的数组操作和数值计算
4. **Matplotlib**：数据可视化工具

### 下一步

- 完成实践练习
- 熟悉 Jupyter Notebook 使用
- 准备进入第 2 章：数学基础

---

<div align="center">

[⬅️ 返回目录](../README.md) | [下一章 ➡️](../chapter02-math-foundations/README.md)

</div>
