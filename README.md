# GRPO Reward Shaping on GSM8K

Comparing Outcome Reward vs Reward Shaping for GRPO training on mathematical reasoning.

## Results

| Model | Train Data | GSM8K pass@1 |
|---|---|---|
| Base Model (no training) | - | 0.175 |
| GRPO + Outcome Reward | 500 samples | 0.315 (+80%) |
| GRPO + Reward Shaping | 500 samples | 0.290 (+66%) |
| GRPO + Outcome Reward | ~5000 samples | **0.610 (+249%)** |

![Reward Comparison](reward_comparison.png)

## Key Findings

1. GRPO training significantly improves mathematical reasoning over the base model.
2. With only 500 training samples, Outcome Reward slightly outperforms Reward Shaping (0.315 vs 0.290). Format/process rewards may introduce noise at small data scales.
3. Scaling training data from 500 to ~5000 samples nearly doubles pass@1 (0.315 → 0.610), demonstrating the importance of data scale for GRPO training.

## Method

**Outcome Reward Only**
- +1.0 if final answer is correct

**Reward Shaping**
- +1.0 if final answer is correct
- +0.2 if output contains `####` format marker
- +0.2 if output contains reasoning steps (arithmetic expressions)

## Setup

- Model: Qwen2.5-1.5B-Instruct
- Dataset: GSM8K
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
