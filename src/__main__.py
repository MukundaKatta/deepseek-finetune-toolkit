"""CLI for deepseek-finetune-toolkit."""
import sys, json, argparse
from .core import DeepseekFinetuneToolkit

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning toolkit for DeepSeek and open-weight LLMs with RLHF support")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = DeepseekFinetuneToolkit()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.process(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"deepseek-finetune-toolkit v0.1.0 — Fine-tuning toolkit for DeepSeek and open-weight LLMs with RLHF support")

if __name__ == "__main__":
    main()
