import re
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = load_from_disk("/root/data/gsm8k")

tokenizer = AutoTokenizer.from_pretrained("/root/models/qwen2.5-1.5b")

def format_prompt(example):
    messages = [
        {"role": "system", "content": "You are a math assistant. Solve step by step. End with '#### <number>'."},
        {"role": "user", "content": example["question"]}
    ]
    return {
        "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        "answer": example["answer"]
    }

dataset = dataset.map(format_prompt)

def extract_answer(text):
    match = re.search(r"####\s*([\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None

def reward_fn(completions, answer, **kwargs):
    rewards = []
    for completion, gt in zip(completions, answer):
        gt_ans = extract_answer(gt)
        pred_ans = extract_answer(completion)
        if gt_ans and pred_ans and gt_ans == pred_ans:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

model = AutoModelForCausalLM.from_pretrained(
    "/root/models/qwen2.5-1.5b",
    torch_dtype="auto",
    device_map="auto"
)

config = GRPOConfig(
    output_dir="/root/autodl-tmp/grpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=512,
    learning_rate=1e-6,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=config,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
)

trainer.train(resume_from_checkpoint="/root/autodl-tmp/grpo_output/checkpoint-2500")
trainer.save_model("./output/final")
print("Done!")
