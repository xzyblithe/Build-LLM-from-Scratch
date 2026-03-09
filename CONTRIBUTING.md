# 贡献指南

感谢你对 **Build LLM from Scratch** 项目的关注！欢迎参与贡献。

## 🤔 如何贡献

### 报告问题

如果你发现了 bug 或有改进建议：

1. 在 [Issues](https://github.com/xzyblithe/Build-LLM-from-Scratch/issues) 中搜索是否已有相关问题
2. 如果没有，创建新 Issue，详细描述问题和复现步骤

### 提交代码

1. **Fork 本仓库**

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **进行修改**
   - 遵循代码风格规范
   - 添加必要的注释
   - 更新相关文档

4. **提交更改**
   ```bash
   git commit -m "feat: 添加新功能描述"
   ```

5. **推送到 Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **创建 Pull Request**

## 📝 代码规范

### Python 代码

```python
# 使用 4 空格缩进
# 函数和类使用 docstring

def example_function(param1: int, param2: str) -> bool:
    """
    函数说明
    
    Args:
        param1: 参数1说明
        param2: 参数2说明
    
    Returns:
        返回值说明
    """
    pass


class ExampleClass:
    """类说明"""
    
    def __init__(self, value: int):
        self.value = value
```

### 文档规范

- 使用 Markdown 格式
- 中文内容使用中文标点
- 代码块指定语言类型

## 🎯 贡献方向

当前需要帮助的方向：

- [ ] 添加单元测试
- [ ] 添加 Jupyter Notebook 示例
- [ ] 完善项目实战代码
- [ ] 添加英文文档
- [ ] 优化代码性能

## 📄 许可证

提交代码即表示你同意你的贡献将按照 [MIT License](LICENSE) 授权。

---

再次感谢你的贡献！🙏
