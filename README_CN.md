# Create Datasets

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white" alt="Windows">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Release-v1.0.0-blue?style=for-the-badge" alt="Release">
</p>

<p align="center">
  <b>LLM 训练数据集生成框架</b><br>
  支持多模型并发生成、断点续传、Web UI 管理
</p>

---

## 简介

Create Datasets 是一个用于生成 SFT/DPO 训练数据的框架，专门针对中文 LLM 微调场景设计。支持多个 LLM API 并发调用，自动进行质量控制和去重，提供 Web 界面和 CLI 两种使用方式。

## 功能特点

- **多模型支持**：Mimo、Grok、DeepSeek 等 OpenAI 兼容 API
- **高并发生成**：支持 30 并发，约 7 条/秒
- **断点续传**：自动生成 checkpoint，中断后可继续
- **质量控制**：自动过滤低质量样本、去重
- **Web UI**：基于 Gradio 的可视化界面
- **灵活配置**：YAML 配置文件，支持模板和种子词

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.9+ | 主语言 |
| httpx | 异步 HTTP 客户端 |
| Gradio | Web UI 框架 |
| PyYAML | 配置文件解析 |
| asyncio | 异步并发 |

## 下载安装

```bash
git clone https://github.com/ygttygtt/Create_datasets.git
cd Create_datasets
pip install -r requirements.txt
```

## 快速开始

### 1. 配置 API

复制环境变量模板并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
MIMO_API_KEY=your_mimo_key
MIMO_API_URL=https://api.example.com/v1
MIMO_MODEL=mimo-v2.5-pro

GROK_API_KEY=your_grok_key
GROK_API_URL=https://api.example.com/v1
GROK_MODEL=grok-4.3-fast
```

### 2. 创建配置文件

复制示例配置：

```bash
cp configs/default.yaml.example configs/my_config.yaml
```

编辑配置文件，设置 API、生成参数、模板等。

### 3. 运行

**Web UI（推荐）：**

```bash
python webui.py
# 浏览器打开 http://127.0.0.1:7860
```

**CLI 模式：**

```bash
# 列出可用配置
python generate.py --list

# 运行生成
python generate.py --config configs/my_config.yaml

# 自定义目标数量
python generate.py --config configs/my_config.yaml --target 500

# 测试模式（不调用 API）
python generate.py --config configs/my_config.yaml --dry-run --target 5
```

## 配置说明

配置文件位于 `configs/*.yaml`，格式如下：

```yaml
name: my_pool
description: "数据池描述"

api:
  base_url: "https://api.example.com/v1"
  key: "YOUR_API_KEY"
  model: "model-name"

generation:
  mode: sft                    # sft 或 dpo
  format: llamafactory         # llamafactory 或 sharegpt
  target: 1000                 # 目标生成数量
  max_concurrent: 10           # 最大并发数
  max_tokens: 2048             # 单次最大 token
  temperature: 0.9             # 温度
  batch_size: 10               # 批量保存大小
  output_dir: "output"         # 输出目录
  output_name: "my_data"       # 输出文件名
  seed: 42                     # 随机种子
  resume: true                 # 是否断点续传

jailbreak:
  enabled: true                # 是否启用破限
  system_prompt: "..."         # 系统提示词
  fake_assistant_reply: "..."  # 预填充回复

templates:
  - name: template1
    weight: 50                 # 采样权重
    system: "系统提示"
    user: "用户提示 {variable}"
    seeds:
      variable: ["选项1", "选项2"]
```

## 输出格式

### SFT 格式

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

## 输出文件

每个配置生成 3 个文件：

| 文件 | 说明 |
|------|------|
| `output/<name>.jsonl` | 有效数据（每行一条） |
| `output/<name>_failures.jsonl` | 失败样本 |
| `output/<name>.checkpoint` | 断点文件 |

## 项目结构

```
Create_Datasets/
├── generate.py              # 主生成脚本
├── webui.py                 # Web 界面
├── configs/                 # 配置文件目录
│   ├── default.yaml.example # 配置示例
│   ├── presets.example.json # 预设示例
│   └── jailbreaks.example.json
├── src/                     # 核心模块
│   ├── config_loader.py     # 配置加载
│   ├── prompt_manager.py    # Prompt 管理
│   ├── generator.py         # 生成器
│   ├── llm_client.py        # API 客户端
│   ├── formatter.py         # 格式转换
│   ├── quality.py           # 质量检查
│   └── rate_limiter.py      # 速率限制
├── tests/                   # 测试
├── docs/                    # 文档
├── output/                  # 输出目录（gitignore）
├── .env.example             # 环境变量模板
├── .gitignore
├── requirements.txt
└── README.md
```

## 安全说明

- API Key 通过环境变量或本地配置文件管理，不会提交到 git
- 生成的数据文件在 `.gitignore` 中，不会自动上传
- 配置文件（含 API Key）在 `.gitignore` 中

## 文档

- [破限指南](docs/jailbreak_guide.md) - Mimo 破限经验和最佳实践

## 许可证

MIT License

## 免责声明

本项目仅用于研究和学习目的。生成的数据内容由 AI 模型产生，不代表开发者立场。使用者应遵守当地法律法规，对使用本项目产生的任何后果自行承担责任。
