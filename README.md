# Deepseek Finetune Toolkit

Fine-tuning toolkit for open-weight LLMs with RLHF

## Features

- Api
Cli
Data Loader
Evaluator
Export
Lora Adapter
Quantizer
Rlhf
Trainer

## Tech Stack

- **Language:** Python
- **Framework:** FastAPI
- **Key Dependencies:** pydantic,fastapi,uvicorn,anthropic,openai,numpy
- **Containerization:** Docker + Docker Compose

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)

### Installation

```bash
git clone https://github.com/MukundaKatta/deepseek-finetune-toolkit.git
cd deepseek-finetune-toolkit
pip install -r requirements.txt
```

### Running

```bash
uvicorn app.main:app --reload
```

### Docker

```bash
docker-compose up
```

## Project Structure

```
deepseek-finetune-toolkit/
├── src/           # Source code
├── tests/         # Test suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## License

MIT
