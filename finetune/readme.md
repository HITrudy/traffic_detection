# Project README.md

## Project Overview
This project contains two main scripts for **trajectory point danger status analysis** and **Qwen2.5 language model fine-tuning**:
- `danger.py`: Processes trajectory point data to calculate acceleration and jerk, then determines danger status and levels based on preset thresholds.
- `finetune.py`: Implements LoRA (Low-Rank Adaptation) based fine-tuning for the Qwen2.5-7B causal language model, supporting custom datasets and training parameters.

# Required Libraries Installation (pip Commands)

```bash
pip install pandas==2.2.2  
pip install datasets==2.19.0 transformers==4.41.2 peft==0.11.1 torch==2.3.1 bitsandbytes==0.43.0 accelerate==0.31.0 tensorboard==2.15.2