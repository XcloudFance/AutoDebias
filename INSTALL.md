# AutoDebias 安装和使用指南

## 📦 安装方法

### 方法一：开发模式安装（推荐）

如果您想要开发或修改代码，建议使用开发模式安装：

```bash
# 进入项目目录
cd autodebias_src

# 开发模式安装
pip install -e .

# 或者安装完整版本（包含所有可选依赖）
pip install -e ".[full]"
```

### 方法二：直接安装

```bash
# 基础安装
pip install .

# 安装特定功能
pip install ".[gpu]"     # GPU加速
pip install ".[dev]"     # 开发工具
pip install ".[full]"    # 完整功能
```

### 方法三：从源码构建

```bash
# 构建分发包
python -m build

# 安装构建的包
pip install dist/autodebias-0.1.0-py3-none-any.whl
```

## 🔧 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0（推荐，用于GPU加速）

## 🚀 验证安装

安装完成后，您可以通过以下方式验证：

### 1. Python导入测试

```python
import autodebias
print(autodebias.__version__)
```

### 2. 命令行测试

```bash
autodebias --help
```

### 3. 功能测试

```python
# 测试基本功能
import autodebias
from diffusers import StableDiffusionPipeline

# 这应该能正常导入而不报错
print("AutoDebias安装成功！")
```

## 📝 使用示例

### 基本使用流程

```python
import autodebias
from diffusers import StableDiffusionPipeline

# 1. 加载模型
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model = model.to("cuda")  # 如果有GPU

# 2. 检测偏见
prompt = "a person working as a doctor"
lookup_table = autodebias.detection(
    model=model, 
    prompt=prompt, 
    num_samples=3,
    detector_type="vlm"  # 可选: "vlm", "openai"
)

# 3. 训练去偏模型
debiased_model = autodebias.debias(
    model=model,
    lookup_table=lookup_table,
    max_training_steps=100
)

# 4. 评估偏见率
results = autodebias.bias_rate(
    model=debiased_model,
    lookup_table=lookup_table,
    prompts=[prompt],
    num_samples=10
)

print("偏见检测和缓解完成！")
```

### 命令行使用

```bash
# 检测偏见
autodebias detect \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --prompt "a person working as a doctor" \
    --num_samples 3 \
    --output bias_lookup.json

# 训练去偏模型
autodebias debias \
    --model_path "runwayml/stable-diffusion-v1-5" \
    --lookup_table bias_lookup.json \
    --steps 100 \
    --output_dir debiased_model

# 评估偏见率
autodebias evaluate \
    --model_path debiased_model \
    --lookup_table bias_lookup.json \
    --prompts "a person working as a doctor" \
    --num_samples 10 \
    --output evaluation_results.json
```

## 🔍 常见问题

### Q: 导入时出现 "No module named 'autodebias'" 错误

**A:** 确保您在正确的虚拟环境中，并且已经安装了包：
```bash
pip list | grep autodebias
```

### Q: GPU内存不足

**A:** 尝试以下解决方案：
- 减少 `num_samples` 参数
- 减少 `max_training_steps` 参数
- 使用较小的模型
- 启用内存优化功能

### Q: OpenAI API 相关错误

**A:** 确保设置了正确的API密钥：
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 🛠️ 开发环境设置

如果您想要贡献代码或开发新功能：

```bash
# 克隆项目
git clone https://github.com/yourusername/autodebias.git
cd autodebias

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black .

# 代码检查
flake8 .
```

## 📊 性能优化建议

1. **GPU使用**: 确保安装了合适的CUDA版本的PyTorch
2. **内存管理**: 定期调用 `torch.cuda.empty_cache()`
3. **批处理**: 适当调整批大小以平衡速度和内存使用
4. **模型量化**: 考虑使用半精度（fp16）训练

## 📞 获取帮助

如果遇到问题，请通过以下方式寻求帮助：

1. 查看 [FAQ 文档](docs/faq.md)
2. 搜索 [GitHub Issues](https://github.com/yourusername/autodebias/issues)
3. 创建新的 Issue 描述您的问题
4. 发送邮件至: autodebias@example.com 