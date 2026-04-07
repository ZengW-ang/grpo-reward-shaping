import re
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = load_from_disk("/root/data/gsm8k")
test_data = dataset["test"].select(range(200))  # 先跑200条

def extract_answer(text):
    match = re.search(r"####\s*([\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None

def evaluate_model(model_path, label):
    print(f"\n评估: {label}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    model.eval()

    correct = 0
    total = len(test_data)

    for i, example in enumerate(test_data):
        messages = [
            {"role": "system", "content": "You are a math assistant. Solve step by step. End with '#### <number>'."},
            {"role": "user", "content": example["question"]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gt_ans = extract_answer(example["answer"])
        pred_ans = extract_answer(completion)

        if gt_ans and pred_ans and gt_ans == pred_ans:
            correct += 1

        if (i+1) % 20 == 0:
            print(f"  {i+1}/{total} | 当前准确率: {correct/(i+1):.3f}")

    acc = correct / total
    print(f"\n{label} 最终 pass@1: {acc:.3f} ({correct}/{total})")
    return acc

# 评估基础模型
acc_base = evaluate_model("/root/models/qwen2.5-1.5b", "Base Model (no training)")

# 评估 Outcome Reward 训练后的模型
acc_outcome = evaluate_model("/root/autodl-tmp/grpo_output/checkpoint-250", "GRPO Outcome Reward")

# 评估 Reward Shaping 训练后的模型
acc_shaped = evaluate_model("/root/autodl-tmp/grpo_shaped_output/final", "GRPO Reward Shaping")

print("\n========== 最终对比 ==========")
print(f"Base Model:      {acc_base:.3f}")
print(f"Outcome Reward:  {acc_outcome:.3f}")
print(f"Reward Shaping:  {acc_shaped:.3f}")
