# 数据集说明

本目录用于存储教程中使用的数据集。

## 📁 数据集列表

### 1. 文本数据

- **shakespeare.txt**：莎士比亚文本，用于语言模型训练
- **zh_text.txt**：中文文本数据
- **dialogues.txt**：对话数据集

### 2. 预训练数据

- **wiki_zh**：中文维基百科数据
- **common_crawl**：Common Crawl 网页数据

### 3. 指令微调数据

- **alpaca_data.json**：Alpaca 指令数据集
- **dolly_data.json**：Dolly 数据集

## 📥 数据获取

部分数据集需要单独下载，请参考各章节说明。

## 📊 数据格式

### 文本数据格式

```
每行一句话
支持 UTF-8 编码
```

### 指令数据格式

```json
{
  "instruction": "指令内容",
  "input": "输入（可选）",
  "output": "期望输出"
}
```

## ⚠️ 注意事项

- 数据文件通常较大，已添加到 .gitignore
- 请根据教程指引下载相应数据集
- 遵守数据集的使用许可协议

## 🔗 常用数据源

- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Common Crawl](https://commoncrawl.org/)
- [Wikipedia Dumps](https://dumps.wikimedia.org/)
