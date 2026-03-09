"""
第12章：Hugging Face Transformers 实战示例
包含 Pipeline、模型加载、微调等
"""
import numpy as np


def pipeline_examples():
    """Pipeline API 示例"""
    code = '''
from transformers import pipeline

# ========== 文本分类 ==========
classifier = pipeline("sentiment-analysis")
result = classifier("I love this movie!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# ========== 命名实体识别 ==========
ner = pipeline("ner", aggregation_strategy="simple")
result = ner("Apple is based in Cupertino, California.")
print(result)

# ========== 文本生成 ==========
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time", max_length=50)
print(result[0]['generated_text'])

# ========== 问答系统 ==========
qa = pipeline("question-answering")
result = qa(
    question="What is the capital of France?",
    context="Paris is the capital and largest city of France."
)
print(result['answer'])

# ========== 文本摘要 ==========
summarizer = pipeline("summarization")
text = "Hugging Face is a company that provides tools..."
result = summarizer(text, max_length=30)
print(result[0]['summary_text'])

# ========== 机器翻译 ==========
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
print(result[0]['translation_text'])
'''
    print("Pipeline API 示例:")
    print(code)


def model_loading_examples():
    """模型加载示例"""
    code = '''
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# ========== 自动加载 ==========
model_name = "bert-base-uncased"

# 加载模型和分词器
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"模型类型: {type(model)}")
print(f"词表大小: {tokenizer.vocab_size}")

# ========== 加载特定任务模型 ==========
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    num_labels=2
)

# ========== 加载语言模型 ==========
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 生成文本
inputs = tokenizer("Hello, I am", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
'''
    print("\n模型加载示例:")
    print(code)


def tokenizer_examples():
    """Tokenizer 示例"""
    code = '''
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
encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
print(f"编码结果: {encoded.keys()}")

# ========== 批处理 ==========
texts = ["Hello world", "How are you?"]
encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print(f"input_ids 形状: {encoded['input_ids'].shape}")

# ========== 句子对 ==========
encoded = tokenizer(
    "Hello world",
    "How are you?",
    padding=True,
    truncation=True,
    return_tensors="pt"
)
print(f"token_type_ids: {encoded['token_type_ids']}")
'''
    print("\nTokenizer 示例:")
    print(code)


def finetuning_examples():
    """微调示例"""
    code = '''
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# ========== 加载数据 ==========
dataset = load_dataset("imdb")

# ========== 加载模型 ==========
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ========== 预处理 ==========
def preprocess(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(preprocess, batched=True)

# ========== 训练参数 ==========
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# ========== 训练器 ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].select(range(1000)),
    eval_dataset=tokenized_dataset["test"].select(range=200)),
)

# 训练
trainer.train()

# 保存
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")
'''
    print("\n微调示例:")
    print(code)


def custom_training_loop():
    """自定义训练循环"""
    code = '''
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
from tqdm import tqdm

# 加载数据
dataset = load_dataset("imdb")

# 加载模型
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 预处理
def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# DataLoader
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
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} 完成")
'''
    print("\n自定义训练循环:")
    print(code)


if __name__ == "__main__":
    print("=" * 60)
    print("Hugging Face Transformers 示例")
    print("=" * 60)
    
    pipeline_examples()
    model_loading_examples()
    tokenizer_examples()
    finetuning_examples()
    custom_training_loop()
    
    print("\n✅ 示例代码展示完成!")
    print("提示: 实际运行需要安装 transformers 和相关依赖")
