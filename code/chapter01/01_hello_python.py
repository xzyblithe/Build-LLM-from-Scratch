"""
第1章代码示例 01：Hello Python
Python 基础语法演示
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
