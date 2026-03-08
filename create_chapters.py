"""
批量创建章节框架脚本
为第5-19章创建基础README和代码目录
"""
import os

# 章节信息
chapters = {
    5: ("Word Embeddings", "词向量与文本表示"),
    6: ("RNN", "循环神经网络"),
    7: ("Attention", "Attention 机制"),
    8: ("Transformer", "Transformer 详解"),
    9: ("Pretraining", "预训练语言模型"),
    10: ("LLM Principles", "大语言模型原理"),
    11: ("PyTorch", "PyTorch 深度学习框架"),
    12: ("Hugging Face", "Hugging Face Transformers 实战"),
    13: ("Transformer from Scratch", "从零实现 Transformer"),
    14: ("GPT from Scratch", "从零实现 GPT"),
    15: ("MoE from Scratch", "从零实现 MoE 架构"),
    16: ("Mainstream LLMs", "从零实现主流大模型"),
    17: ("PEFT", "参数高效微调"),
    18: ("Instruction Tuning & RLHF", "指令微调与 RLHF"),
    19: ("Deployment", "模型部署与推理优化")
}

# 模板
template = """# 第{num}章：{cn_title}

<div align="center">

[⬅️ 上一章](../chapter{prev:02d}-{prev_folder}/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter{next:02d}-{next_folder}/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解{cn_title}的核心概念
- ✅ 掌握相关技术和实现方法
- ✅ 通过实践项目加深理解

---

## 🎯 本章内容

### 1. 核心概念

（内容编写中...）

### 2. 技术原理

（内容编写中...）

### 3. 实践应用

（内容编写中...）

---

## 💻 代码示例

参见 `code/chapter{num:02d}/` 目录。

---

## 🎯 实践练习

### 练习 1

（待补充）

### 练习 2

（待补充）

---

## 📝 本章小结

### 核心要点

1. 要点 1
2. 要点 2
3. 要点 3

### 关键概念

- 概念 1
- 概念 2

---

<div align="center">

[⬅️ 上一章](../chapter{prev:02d}-{prev_folder}/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter{next:02d}-{next_folder}/README.md)

</div>
"""

# 创建章节文件
chapter_list = list(chapters.items())

for i, (num, (en_title, cn_title)) in enumerate(chapter_list):
    # 获取前后章节信息
    prev_num = num - 1 if num > 1 else 1
    next_num = num + 1 if num < 19 else 19
    
    prev_folder = chapters.get(prev_num, ("", ""))[0].lower().replace(" ", "-")
    next_folder = chapters.get(next_num, ("", ""))[0].lower().replace(" ", "-")
    
    # 生成内容
    content = template.format(
        num=num,
        cn_title=cn_title,
        prev=prev_num,
        next=next_num,
        prev_folder=prev_folder,
        next_folder=next_folder
    )
    
    # 写入文件
    chapter_dir = f"chapters/chapter{num:02d}-{en_title.lower().replace(' ', '-')}"
    os.makedirs(chapter_dir, exist_ok=True)
    
    readme_path = f"{chapter_dir}/README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 创建代码目录
    code_dir = f"code/chapter{num:02d}"
    os.makedirs(code_dir, exist_ok=True)
    
    # 创建占位文件
    placeholder = f'"""\n第{num}章代码示例\n{cn_title}\n"""\n# 内容编写中...\n'
    with open(f"{code_dir}/placeholder.py", 'w', encoding='utf-8') as f:
        f.write(placeholder)
    
    print(f"创建章节 {num}: {cn_title}")

print("\n所有章节框架已创建完成！")
