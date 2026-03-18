# deepseek-finetune-toolkit

**Fine-tuning toolkit for DeepSeek and open-weight LLMs with RLHF support**

![Build](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-proprietary-red)

## Install
```bash
pip install -e ".[dev]"
```

## Quick Start
```python
from src.core import DeepseekFinetuneToolkit
 instance = DeepseekFinetuneToolkit()
r = instance.process(input="test")
```

## CLI
```bash
python -m src status
python -m src run --input "data"
```

## API
| Method | Description |
|--------|-------------|
| `process()` | Process |
| `analyze()` | Analyze |
| `transform()` | Transform |
| `validate()` | Validate |
| `export()` | Export |
| `get_stats()` | Get stats |
| `get_stats()` | Get stats |
| `reset()` | Reset |

## Test
```bash
pytest tests/ -v
```

