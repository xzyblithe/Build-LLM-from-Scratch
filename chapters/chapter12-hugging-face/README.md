# 第12章：Hugging Face Transformers 实战

<div align="center">

[⬅️ 上一章](../chapter11-pytorch/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter13-transformer-from-scratch/README.md)

</div>

---

## 📖 学习目标

学完本章后，你将能够：

- ✅ 理解 Hugging Face 生态系统
- ✅ 使用 Transformers 库加载和微调预训练模型
- ✅ 掌握 Tokenizer 的使用方法
- ✅ 使用 Datasets 库处理大规模数据
- ✅ 实现文本分类、命名实体识别等任务

---

## 🎯 本章内容

### 1. Hugging Face 简介

#### 1.1 生态系统概览

```
Hugging Face 生态系统：

┌─────────────────────────────────────────────────────┐
│                  Hugging Face Hub                    │
│  (模型仓库、数据集仓库、Spaces、Model Cards)          │
└─────────────────────────────────────────────────────┘
           │           │            │
           ▼           ▼            ▼
┌──────────────┐ ┌──────────┐ ┌─────────────┐
│ Transformers │ │ Datasets │ │ Tokenizers  │
│   模型库      │ │  数据集   │ │   分词器     │
└──────────────┘ └──────────┘ └─────────────┘
           │           │            │
           ▼           ▼            ▼
┌──────────────────────────────────────────────────────┐
│                     Accelerate                        │
│            (分布式训练、混合精度)                       │
└──────────────────────────────────────────────────────┘
```

#### 1.2 安装与配置

```bash
# 安装核心库
pip install transformers datasets tokenizers accelerate

# 安装深度学习框架（二选一）
pip install torch          # PyTorch
pip install tensorflow     # TensorFlow

# 安装额外工具
pip install evaluate       # 评估指标
pip install peft           # 参数高效微调
pip install bitsandbytes   # 量化支持
```

---

### 2. Transformers 快速入门

#### 2.1 Pipeline API

```python
from transformers import pipeline

# ========== 文本分类 ==========
classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
print(f"结果: {result}")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# ========== 文本生成 ==========
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)
print(f"生成文本: {result[0]['generated_text']}")

# ========== 问答系统 ==========
qa = pipeline("question-answering")
result = qa(
    question="What is the capital of France?",
    context="Paris is the capital and largest city of France."
)
print(f"答案: {result['answer']}")

# ========== 命名实体识别 ==========
ner = pipeline("ner", aggregation_strategy="simple")
result = ner("Apple is based in Cupertino, California.")
print(f"实体: {result}")

# ========== 文本摘要 ==========
summarizer = pipeline("summarization")
text = """
Hugging Face is a company that provides tools for natural language processing.
They created the Transformers library, which has become the de facto standard
for working with pretrained language models.
"""
result = summarizer(text, max_length=30)
print(f"摘要: {result[0]['summary_text']}")

# ========== 机器翻译 ==========
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
print(f"翻译: {result[0]['translation_text']}")
```

#### 2.2 加载预训练模型

```python
from transformers import AutoModel, AutoTokenizer

# 加载模型和分词器
model_name = "bert-base-uncased"

# 方式1：自动加载
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"模型类型: {type(model)}")
print(f"词表大小: {tokenizer.vocab_size}")

# 方式2：加载特定类型模型
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 查看模型结构
print(model)
```

---

### 3. Tokenizer 详解

#### 3.1 分词器类型

```
主流分词算法：

1. WordPiece（BERT）
   - 基于字符的子词分词
   - 例如："unhappy" → ["un", "##happy"]

2. BPE（Byte Pair Encoding）（GPT）
   - 基于字节对合并
   - 例如："lower" → ["low", "er"]

3. SentencePiece（T5, ALBERT）
   - 语言无关的分词
   - 直接处理原始文本

4. Unigram（XLNet）
   - 基于概率的语言模型分词
```

#### 3.2 使用 Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ========== 基本分词 ==========
text = "Hello, how are you?"

# 分词
tokens = tokenizer.tokenize(text)
print(f"分词结果: {tokens}")
# ['hello', ',', 'how', 'are', 'you', '?']

# 转换为索引
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"索引: {ids}")

# 解码
decoded = tokenizer.decode(ids)
print(f"解码: {decoded}")

# ========== 编码（一步到位）==========
# 单个文本
encoded = tokenizer(text)
print(f"\n编码结果: {encoded}")
# {'input_ids': [...], 'token_type_ids': [...], 'attention_mask': [...]}

# 多个文本
texts = ["Hello world", "How are you?"]
encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print(f"批处理编码:\n{encoded}")

# ========== 特殊参数 ==========
encoded = tokenizer(
    text,
    padding="max_length",      # 填充到最大长度
    max_length=20,             # 最大长度
    truncation=True,           # 超长截断
    return_tensors="pt",       # 返回 PyTorch 张量
    return_attention_mask=True,
    return_token_type_ids=True
)

print(f"\ninput_ids 形状: {encoded['input_ids'].shape}")
print(f"attention_mask 形状: {encoded['attention_mask'].shape}")
```

#### 3.3 处理特殊标记

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 特殊标记
print(f"CLS 标记: {tokenizer.cls_token} (id: {tokenizer.cls_token_id})")
print(f"SEP 标记: {tokenizer.sep_token} (id: {tokenizer.sep_token_id})")
print(f"PAD 标记: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
print(f"UNK 标记: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")

# 处理句子对
sentence1 = "Hello world"
sentence2 = "How are you?"

encoded = tokenizer(
    sentence1,
    sentence2,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

print(f"\n句子对编码:")
print(f"input_ids: {encoded['input_ids']}")
print(f"token_type_ids: {encoded['token_type_ids']}")  # 区分两个句子
```

---

### 4. 模型微调

#### 4.1 文本分类微调

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("imdb")
print(f"训练集: {len(dataset['train'])} 样本")
print(f"测试集: {len(dataset['test'])} 样本")

# 加载模型和分词器
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# 预处理数据
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["text"]
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),  # 小样本演示
    eval_dataset=tokenized_datasets["test"].select(range(200)),
)

# 训练
trainer.train()

# 评估
eval_result = trainer.evaluate()
print(f"评估结果: {eval_result}")

# 保存模型
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")
```

#### 4.2 使用原生 PyTorch 微调

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
from tqdm import tqdm

# 加载数据
dataset = load_dataset("imdb")

# 加载模型和分词器
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 数据预处理
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 数据加载器
train_loader = DataLoader(
    tokenized_dataset["train"].select(range(1000)),
    batch_size=8,
    shuffle=True
)

# 设备和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# 训练循环
model.train()
for epoch in range(2):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

print("训练完成!")
```

---

### 5. Datasets 库

#### 5.1 加载数据集

```python
from datasets import load_dataset

# 加载 Hub 数据集
dataset = load_dataset("imdb")
print(dataset)

# 加载本地数据
# CSV
dataset = load_dataset("csv", data_files="my_data.csv")

# JSON
dataset = load_dataset("json", data_files="my_data.json")

# 文本文件
dataset = load_dataset("text", data_files="my_data.txt")

# 加载特定子集
dataset = load_dataset("glue", "mrpc")
```

#### 5.2 数据处理

```python
from datasets import load_dataset

dataset = load_dataset("imdb")

# ========== 选择和过滤 ==========
# 选择部分数据
small_dataset = dataset["train"].select(range(100))

# 过滤数据
filtered = dataset["train"].filter(lambda x: len(x["text"]) > 1000)
print(f"过滤后: {len(filtered)} 样本")

# ========== 映射处理 ==========
# 添加新列
def add_length(example):
    example["text_length"] = len(example["text"])
    return example

dataset = dataset.map(add_length)

# 批量处理
def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], padding=True, truncation=True)

# ========== 数据集操作 ==========
# 打乱
shuffled = dataset["train"].shuffle(seed=42)

# 划分
split = dataset["train"].train_test_split(test_size=0.2)
print(f"训练集: {len(split['train'])}, 测试集: {len(split['test'])}")

# 拼接
combined = dataset["train"].concatenate(dataset["test"])
```

---

### 6. 常见任务实战

#### 6.1 命名实体识别（NER）

```python
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# 使用 pipeline
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

text = "Apple Inc. is headquartered in Cupertino, California. Tim Cook is the CEO."
results = ner(text)

print("命名实体识别结果:")
for result in results:
    print(f"  {result['word']}: {result['entity_group']} (score: {result['score']:.4f})")

# 自定义模型微调
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 加载 NER 数据集
dataset = load_dataset("conll2003")
print(f"标签: {dataset['train'].features['ner_tags'].feature.names}")
```

#### 6.2 问答系统

```python
from transformers import pipeline

# 使用 pipeline
qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = """
The Great Wall of China is a series of fortifications made of stone, brick, 
tamped earth, wood, and other materials. It was built along the historical 
northern borders of China. The wall was built to protect the Chinese states 
from various nomadic groups.
"""

questions = [
    "What is the Great Wall made of?",
    "Why was the Great Wall built?",
    "Where is the Great Wall located?"
]

print("问答结果:")
for question in questions:
    result = qa(question=question, context=context)
    print(f"  Q: {question}")
    print(f"  A: {result['answer']} (score: {result['score']:.4f})\n")
```

#### 6.3 文本生成

```python
from transformers import pipeline, set_seed

# GPT-2 文本生成
generator = pipeline("text-generation", model="gpt2")
set_seed(42)

prompt = "In the future, artificial intelligence will"
result = generator(
    prompt,
    max_length=100,
    num_return_sequences=3,
    temperature=0.7,
    do_sample=True
)

print("生成的文本:")
for i, text in enumerate(result):
    print(f"\n[{i+1}] {text['generated_text']}")

# 使用不同的解码策略
result_beam = generator(
    prompt,
    max_length=100,
    num_beams=5,
    num_return_sequences=3,
    early_stopping=True
)

print("\n束搜索生成:")
for i, text in enumerate(result_beam):
    print(f"\n[{i+1}] {text['generated_text']}")
```

---

### 7. 模型量化与加速

#### 7.1 动态量化

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 比较大小
def print_size(model, name):
    torch.save(model.state_dict(), "temp.pt")
    size = os.path.getsize("temp.pt") / (1024 * 1024)
    print(f"{name}: {size:.2f} MB")
    os.remove("temp.pt")

import os
print_size(model, "原始模型")
print_size(quantized_model, "量化模型")
```

#### 7.2 使用 bitsandbytes 量化

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

print(f"模型加载成功，显存占用大幅减少")
```

---

## 💻 完整代码示例

### 示例：情感分析应用

```python
"""
完整的情感分析应用
包括数据加载、模型微调、评估和部署
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset, Dataset
import numpy as np
from evaluate import load

class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_map = {0: "负面", 1: "正面"}
    
    def prepare_data(self, texts, labels=None):
        """准备数据"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        if labels is not None:
            encoded["labels"] = torch.tensor(labels)
        
        return encoded
    
    def train(self, train_texts, train_labels, eval_texts=None, eval_labels=None,
              epochs=3, batch_size=8):
        """训练模型"""
        # 初始化模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        
        # 创建数据集
        train_dataset = Dataset.from_dict({
            "text": train_texts,
            "label": train_labels
        })
        
        # 预处理
        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        train_dataset = train_dataset.map(tokenize_fn, batched=True)
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir="./sentiment_model",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="epoch",
        )
        
        # 训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        print("训练完成!")
    
    def predict(self, texts):
        """预测情感"""
        self.model.eval()
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            predictions = torch.softmax(outputs.logits, dim=-1)
            labels = torch.argmax(predictions, dim=-1)
        
        results = []
        for i, text in enumerate(texts):
            results.append({
                "text": text,
                "label": self.label_map[labels[i].item()],
                "score": predictions[i][labels[i]].item()
            })
        
        return results
    
    def save(self, path):
        """保存模型"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"模型已保存到 {path}")
    
    def load(self, path):
        """加载模型"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"模型已从 {path} 加载")


# 使用示例
if __name__ == "__main__":
    # 示例数据
    train_texts = [
        "这部电影太好看了！",
        "非常失望，浪费时间。",
        "我非常喜欢这个故事。",
        "太糟糕了，不推荐。",
        "演员表演很出色。",
        "剧情无聊，没有新意。"
    ]
    train_labels = [1, 0, 1, 0, 1, 0]  # 1: 正面, 0: 负面
    
    # 创建分析器并训练
    analyzer = SentimentAnalyzer()
    
    # 预测（使用预训练模型）
    test_texts = ["这部电影真的很好看！", "太无聊了"]
    results = analyzer.predict(test_texts)
    
    print("预测结果:")
    for result in results:
        print(f"  文本: {result['text']}")
        print(f"  情感: {result['label']}")
        print(f"  置信度: {result['score']:.4f}\n")
```

---

## 🎯 实践练习

### 练习 1：构建问答系统

```python
# TODO: 使用 SQuAD 数据集微调问答模型
# 1. 加载数据集
# 2. 预处理数据
# 3. 微调模型
# 4. 评估性能
```

### 练习 2：文本摘要微调

```python
# TODO: 使用 CNN/DailyMail 数据集微调摘要模型
# 1. 加载数据集
# 2. 预处理
# 3. 训练 BART 或 T5
# 4. 生成摘要并评估 ROUGE 分数
```

---

## 📝 本章小结

### 核心要点

1. **Pipeline**：快速使用预训练模型的最简单方式
2. **Tokenizer**：文本与模型输入之间的桥梁
3. **AutoModel**：自动加载任意架构的预训练模型
4. **Trainer**：简化训练循环的高级 API
5. **Datasets**：高效处理大规模数据集

### 关键概念

- 预训练模型（Pre-trained Model）
- 微调（Fine-tuning）
- 子词分词（Subword Tokenization）
- 注意力掩码（Attention Mask）
- 模型量化（Quantization）

---

<div align="center">

[⬅️ 上一章](../chapter11-pytorch/README.md) | [返回目录](../README.md) | [下一章 ➡️](../chapter13-transformer-from-scratch/README.md)

</div>
