"""
第10章：大语言模型原理
包括 Tokenizer、模型架构、生成策略
"""
import numpy as np
from typing import List, Optional, Dict
from collections import Counter


class SimpleTokenizer:
    """简单的字符级分词器"""
    
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
    
    def train(self, corpus: List[str], vocab_size: int = 1000):
        """从语料训练词表"""
        # 统计字符频率
        char_freq = Counter()
        for text in corpus:
            char_freq.update(text)
        
        # 选择最频繁的字符
        most_common = char_freq.most_common(vocab_size - 4)  # 预留特殊 token
        
        # 构建词表
        self.vocab = {self.pad_token: 0, self.unk_token: 1, 
                      self.bos_token: 2, self.eos_token: 3}
        for i, (char, _) in enumerate(most_common):
            self.vocab[char] = i + 4
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本"""
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.vocab[self.bos_token])
        
        for char in text:
            tokens.append(self.vocab.get(char, self.vocab[self.unk_token]))
        
        if add_special_tokens:
            tokens.append(self.vocab[self.eos_token])
        
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """解码 token"""
        special_ids = {self.vocab[self.pad_token], self.vocab[self.bos_token],
                       self.vocab[self.eos_token]}
        
        chars = []
        for token in tokens:
            if skip_special_tokens and token in special_ids:
                continue
            chars.append(self.inv_vocab.get(token, self.unk_token))
        
        return ''.join(chars)


class BPETokenizer:
    """Byte Pair Encoding 分词器（简化版）"""
    
    def __init__(self):
        self.merges = []
        self.vocab = {}
    
    def train(self, corpus: List[str], num_merges: int = 1000):
        """训练 BPE"""
        # 初始化：每个字符是一个 token
        vocab = Counter()
        for text in corpus:
            for char in text:
                vocab[char] += 1
        
        # 迭代合并
        for _ in range(num_merges):
            # 找最频繁的相邻 pair
            pairs = Counter()
            for text in corpus:
                tokens = list(text)
                for i in range(len(tokens) - 1):
                    pairs[(tokens[i], tokens[i+1])] += 1
            
            if not pairs:
                break
            
            best_pair = pairs.most_common(1)[0][0]
            
            # 合并
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            
            # 更新语料
            new_corpus = []
            for text in corpus:
                new_text = text.replace(''.join(best_pair), new_token)
                new_corpus.append(new_text)
            corpus = new_corpus
        
        # 构建词表
        for text in corpus:
            for token in text:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)


class GenerationConfig:
    """生成配置"""
    
    def __init__(self,
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 1.0,
                 do_sample: bool = True,
                 repetition_penalty: float = 1.0):
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty


class LLMGenerator:
    """大语言模型生成器"""
    
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 简化的模型参数
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
    
    def generate(self, input_ids: List[int], 
                 config: GenerationConfig) -> List[int]:
        """
        生成文本
        
        参数:
            input_ids: 输入 token IDs
            config: 生成配置
        
        返回:
            生成的 token IDs
        """
        tokens = list(input_ids)
        
        for _ in range(config.max_length):
            # 获取 logits（简化）
            logits = self._get_logits(tokens[-1])
            
            # 应用惩罚
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, tokens, config.repetition_penalty
                )
            
            # 温度
            logits = logits / config.temperature
            
            # 采样
            if config.do_sample:
                next_token = self._sample(logits, config.top_k, config.top_p)
            else:
                next_token = int(np.argmax(logits))
            
            tokens.append(next_token)
            
            # EOS 检查
            if next_token == 3:  # </s>
                break
        
        return tokens
    
    def _get_logits(self, token_id: int) -> np.ndarray:
        """获取 logits（简化）"""
        hidden = self.embedding[token_id]
        return np.matmul(hidden, self.lm_head)
    
    def _apply_repetition_penalty(self, logits: np.ndarray, 
                                   tokens: List[int], 
                                   penalty: float) -> np.ndarray:
        """应用重复惩罚"""
        for token in set(tokens):
            if logits[token] > 0:
                logits[token] /= penalty
            else:
                logits[token] *= penalty
        return logits
    
    def _sample(self, logits: np.ndarray, top_k: int, top_p: float) -> int:
        """采样"""
        # Top-K
        if top_k > 0:
            top_k_indices = np.argsort(logits)[-top_k:]
            mask = np.ones(self.vocab_size, dtype=bool)
            mask[top_k_indices] = False
            logits[mask] = -float('inf')
        
        # Top-P
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        probs = np.exp(sorted_logits - np.max(sorted_logits))
        probs = probs / probs.sum()
        
        cumulative_probs = np.cumsum(probs)
        top_p_mask = cumulative_probs <= top_p
        if not any(top_p_mask):
            top_p_mask[0] = True
        
        candidates = sorted_indices[top_p_mask]
        probs_p = probs[top_p_mask]
        probs_p = probs_p / probs_p.sum()
        
        return int(np.random.choice(candidates, p=probs_p))


class PromptTemplate:
    """提示模板"""
    
    @staticmethod
    def instruction_format(instruction: str, input_text: str = "") -> str:
        """指令格式"""
        if input_text:
            return f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 输出:\n"
        return f"### 指令:\n{instruction}\n\n### 输出:\n"
    
    @staticmethod
    def chat_format(messages: List[Dict]) -> str:
        """对话格式"""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted += f"User: {content}\n"
            else:
                formatted += f"Assistant: {content}\n"
        formatted += "Assistant:"
        return formatted


def test_llm_principles():
    """测试大语言模型原理"""
    print("=" * 60)
    print("大语言模型原理测试")
    print("=" * 60)
    
    # ========== Tokenizer ==========
    print("\n【Tokenizer】")
    tokenizer = SimpleTokenizer()
    corpus = ["你好世界", "我喜欢学习人工智能", "大语言模型很有趣"]
    tokenizer.train(corpus, vocab_size=100)
    
    text = "你好"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    print(f"原文: {text}")
    print(f"编码: {tokens}")
    print(f"解码: {decoded}")
    print(f"词表大小: {len(tokenizer.vocab)}")
    
    # ========== 生成策略 ==========
    print("\n【生成策略】")
    generator = LLMGenerator(vocab_size=100, d_model=64)
    
    configs = [
        ("Greedy", GenerationConfig(do_sample=False, max_length=20)),
        ("Top-K (k=5)", GenerationConfig(top_k=5, temperature=0.8, max_length=20)),
        ("Top-P (p=0.9)", GenerationConfig(top_p=0.9, temperature=0.8, max_length=20)),
    ]
    
    for name, config in configs:
        input_ids = [2, 10, 20]  # BOS + 一些 token
        output = generator.generate(input_ids, config)
        print(f"{name}: 输入长度={len(input_ids)}, 输出长度={len(output)}")
    
    # ========== 提示模板 ==========
    print("\n【提示模板】")
    instruction = "翻译下面的句子"
    input_text = "Hello World"
    prompt = PromptTemplate.instruction_format(instruction, input_text)
    print(f"指令模板:\n{prompt[:100]}...")
    
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
        {"role": "user", "content": "介绍一下自己"}
    ]
    chat_prompt = PromptTemplate.chat_format(messages)
    print(f"\n对话模板:\n{chat_prompt[:100]}...")
    
    print("\n✅ 大语言模型原理测试通过!")


if __name__ == "__main__":
    test_llm_principles()
