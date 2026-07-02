# Create Datasets

一个用于生成 SFT/DPO 训练数据的框架，专门针对中文 LLM 微调场景。

## 🎯 项目特点

- **多模型支持**：Mimo、Grok、DeepSeek 等多个 LLM API
- **高并发生成**：支持 30 并发，~7 条/秒
- **断点续传**：自动生成 checkpoint，中断后可继续
- **质量控制**：自动过滤低质量样本、去重
- **Web UI**：基于 Gradio 的可视化界面
- **多种数据类型**：NSFW、通用对话、脏话、自然聊天等

## 📁 项目结构

```
├── generate.py          # 主生成脚本
├── webui.py             # Web 界面
├── configs/             # 配置文件（需要自己创建）
│   └── *.example.json   # 示例配置
├── src/                 # 核心模块
│   ├── config_loader.py
│   ├── prompt_manager.py
│   ├── generator.py
│   ├── llm_client.py
│   └── ...
├── output/              # 生成的数据（gitignore）
├── scripts/             # 辅助脚本
│   ├── tests/           # 测试脚本
│   └── legacy/          # 遗留脚本
├── docs/                # 文档
└── .env.example         # 环境变量示例
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API

复制 `.env.example` 为 `.env`，填入你的 API key：

```bash
cp .env.example .env
# 编辑 .env 填入你的真实 API key
```

### 3. Web UI（推荐）

```bash
python webui.py
# 浏览器打开 http://127.0.0.1:7860
```

### 4. CLI 模式

```bash
# 列出可用配置
python generate.py --list

# 运行生成
python generate.py --config configs/nsfw_mimo.yaml

# 自定义目标数量
python generate.py --config configs/nsfw_mimo.yaml --target 500

# 测试模式（不调用 API）
python generate.py --config configs/nsfw_mimo.yaml --dry-run --target 5
```

## 📊 数据集格式

### SFT 格式（推荐）

```json
{
  "instruction": "用户的问题或指令",
  "input": "可选的额外输入",
  "output": "模型的回复",
  "system": "系统提示（可选）",
  "history": []
}
```

### DPO 格式

```json
{
  "system": "系统提示",
  "instruction": "用户指令",
  "input": "",
  "chosen": "偏好的回复",
  "rejected": "不偏好的回复"
}
```

## 🔧 配置文件

配置文件位于 `configs/*.yaml`，格式如下：

```yaml
name: my_pool
description: "数据池描述"

api:
  base_url: "https://api.example.com/v1"
  key: "YOUR_API_KEY"  # 或使用环境变量
  model: "model-name"

generation:
  mode: sft
  format: llamafactory
  target: 1000
  max_concurrent: 10
  max_tokens: 2048
  temperature: 0.9
  batch_size: 10
  output_dir: "output"
  output_name: "my_data"
  seed: 42

jailbreak:
  enabled: true
  system_prompt: "破限提示词..."
  fake_assistant_reply: "好的回复..."

templates:
  - name: template1
    weight: 50
    system: "系统提示"
    user: "用户提示 {variable}"
    seeds:
      variable: ["选项1", "选项2"]
```

## 📈 输出文件

每个配置生成 3 个文件：

| 文件 | 说明 |
|------|------|
| `output/<name>.jsonl` | 有效数据（每行一条） |
| `output/<name>_failures.jsonl` | 失败样本 |
| `output/<name>.checkpoint` | 断点文件 |

## 🎓 训练建议

### Qwen3-3B 微调（推荐入门）

```
模型: Qwen3-3B-Instruct
显卡: RTX 4090 24GB
方法: QLoRA (4-bit)
VRAM: ~6GB
数据: 2000-3000 条
训练时间: 1-2 小时
```

### Qwen3-14B 微调（推荐进阶）

```
模型: Qwen3-14B-Instruct
显卡: RTX 4090 24GB 或 5090 32GB
方法: QLoRA (4-bit)
VRAM: ~12GB
数据: 3000-5000 条
训练时间: 3-6 小时
```

## 🔐 安全说明

- **API Key**：所有 API key 通过环境变量或本地配置文件管理，不会提交到 git
- **输出数据**：生成的数据文件在 `.gitignore` 中，不会自动上传
- **配置文件**：`configs/` 目录在 `.gitignore` 中

## 📚 文档

- [破限指南](docs/jailbreak_guide.md) — Mimo 破限经验和最佳实践
- [模型对比](MODEL_COMPARISON.md) — DeepSeek、Grok、Mimo 对比
- [数据集管理](scripts/merge_and_validate.py) — 数据清洗和合并工具

## 🛠️ 辅助脚本

```bash
# 数据集管理
python scripts/merge_and_validate.py    # 合并和验证数据
python scripts/clean_data.py            # 清洗数据
python scripts/nsfw_dataset.py          # NSFW 数据分类和抽取

# 测试
python scripts/tests/test_mimo_quick.py # 快速测试 Mimo
```

## 📝 更新日志

- **2026-07-02**: 数据集清洗完成，支持 Qwen3 微调
- **2026-07-01**: 初始版本，支持多模型生成

## 📄 许可证

MIT License

## ⚠️ 免责声明

本项目仅用于研究和学习目的。生成的数据内容由 AI 模型产生，不代表开发者立场。使用者应遵守当地法律法规，对使用本项目产生的任何后果自行承担责任。
