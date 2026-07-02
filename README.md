# Create Datasets

A framework for generating SFT/DPO training data, designed for Chinese LLM fine-tuning scenarios.

## Features

- **Multi-model support**: Mimo, Grok, DeepSeek, and other OpenAI-compatible APIs
- **High concurrency**: Up to 30 concurrent requests, ~7 samples/second
- **Checkpoint resume**: Automatic checkpoint saving, resume from interruption
- **Quality control**: Auto-filter low-quality samples, deduplication
- **Web UI**: Gradio-based visual interface
- **Multiple data types**: Configurable templates with weighted sampling

## Project Structure

```
├── generate.py          # Main generation script
├── webui.py             # Web interface
├── configs/             # Configuration files (create your own)
│   └── *.example.json   # Example configurations
├── src/                 # Core modules
│   ├── config_loader.py
│   ├── prompt_manager.py
│   ├── generator.py
│   ├── llm_client.py
│   └── ...
├── output/              # Generated data (gitignored)
├── docs/                # Documentation
└── .env.example         # Environment variables template
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
# Edit .env with your real API keys
```

### 3. Web UI (Recommended)

```bash
python webui.py
# Open http://127.0.0.1:7860 in browser
```

### 4. CLI Mode

```bash
# List available configs
python generate.py --list

# Run generation
python generate.py --config configs/my_config.yaml

# Custom target count
python generate.py --config configs/my_config.yaml --target 500

# Test mode (no API calls)
python generate.py --config configs/my_config.yaml --dry-run --target 5
```

## Data Format

### SFT Format (Recommended)

```json
{
  "instruction": "User question or instruction",
  "input": "Optional additional input",
  "output": "Model response",
  "system": "System prompt (optional)",
  "history": []
}
```

### DPO Format

```json
{
  "system": "System prompt",
  "instruction": "User instruction",
  "input": "",
  "chosen": "Preferred response",
  "rejected": "Non-preferred response"
}
```

## Configuration

Configuration files are located in `configs/*.yaml`. Format:

```yaml
name: my_pool
description: "Data pool description"

api:
  base_url: "https://api.example.com/v1"
  key: "YOUR_API_KEY"  # Or use environment variables
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
  system_prompt: "Your jailbreak prompt..."
  fake_assistant_reply: "Assistant reply..."

templates:
  - name: template1
    weight: 50
    system: "System prompt"
    user: "User prompt with {variable}"
    seeds:
      variable: ["option1", "option2"]
```

## Output Files

Each configuration generates 3 files:

| File | Description |
|------|-------------|
| `output/<name>.jsonl` | Valid samples (one per line) |
| `output/<name>_failures.jsonl` | Failed attempts |
| `output/<name>.checkpoint` | Checkpoint file |

## Security

- **API Keys**: Managed via environment variables or local config files, never committed to git
- **Output Data**: Generated data files are in `.gitignore`
- **Config Files**: `configs/` directory is in `.gitignore`

## Documentation

- [Jailbreak Guide](docs/jailbreak_guide.md) - Mimo jailbreak experience and best practices

## License

MIT License

## Disclaimer

This project is for research and educational purposes only. Generated content is produced by AI models and does not represent the developer's views. Users should comply with local laws and regulations.
