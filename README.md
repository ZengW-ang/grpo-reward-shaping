# GRPO Reward Shaping on GSM8K

Comparing **Outcome Reward** vs **Reward Shaping** for GRPO training on mathematical reasoning.

## Results

| Model | GSM8K pass@1 |
|---|---|
| Base Model (no training) | 0.175 |
| GRPO + Outcome Reward | **0.315** (+80%) |
| GRPO + Reward Shaping | 0.290 (+66%) |

![Reward Comparison](reward_comparison.png)

## Key Finding

Both GRPO variants significantly outperform the base model. With only 500 training samples, Outcome Reward slightly outperforms Reward Shaping on pass@1 (0.315 vs 0.290). We hypothesize that with limited data, format/process rewards introduce noise that outweighs their benefit — consistent with DeepSeek-R1's observation that reward shaping requires sufficient data scale to stabilize.

## Method

**Outcome Reward Only**
- +1.0 if final answer is correct

**Reward Shaping**
- +1.0 if final answer is correct
- +0.2 if output contains `####` format marker
- +0.2 if output contains reasoning steps (arithmetic expressions)

## Setup

- Model: Qwen2.5-1.5B-Instruct
- Dataset: GSM8K (500 train / 200 test)
- Algorithm: GRPO via `trl`
- Hardware: A800 80G

## Install
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers trl datasets accelerate
```

## Run
```bash
python train_outcome_reward.py
python train_reward_shaping.py
python evaluate.py
python plot_results.py
```
