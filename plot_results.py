import matplotlib.pyplot as plt

steps = list(range(10, 260, 10))

outcome_rewards = [0.15, 0.3375, 0.225, 0.325, 0.1625, 0.3625, 0.375, 0.3,
                   0.3, 0.2875, 0.3625, 0.2375, 0.325, 0.3125, 0.3125, 0.2875,
                   0.2875, 0.375, 0.2625, 0.3, 0.3125, 0.275, 0.375, 0.35, 0.3125]

shaped_rewards = [0.395, 0.58, 0.4425, 0.59, 0.39, 0.535, 0.5925, 0.5375,
                  0.5375, 0.4675, 0.5675, 0.6025, 0.5025, 0.49, 0.675, 0.45,
                  0.6725, 0.6025, 0.6, 0.5575, 0.51, 0.68, 0.6625, 0.6225, 0.73]

plt.figure(figsize=(10, 6))
plt.plot(steps, outcome_rewards, label='Outcome Reward Only', marker='o', markersize=3)
plt.plot(steps, shaped_rewards, label='Reward Shaping', marker='s', markersize=3)
plt.xlabel('Training Steps')
plt.ylabel('Reward')
plt.title('GRPO Training: Outcome Reward vs Reward Shaping on GSM8K')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/root/autodl-tmp/reward_comparison.png', dpi=150)
print("图已保存！")
