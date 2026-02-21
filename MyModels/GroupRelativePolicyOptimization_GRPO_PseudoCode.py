import torch
import torch.nn.functional as F

# constants
kl_beta = 0.1
eps = 0.2

# sample G completions for B prompts
# compute outcome reward for each completion
with torch.no_grad():
    completions = LLM.generate(prompts)  # (B*G, L)
    rewards = RM(completions)  # (B*G)

# create a padding mask from lengths of completions in batch
completion_mask = <... mask out padding tokens ...>

# get policy logprobs for each action
llm_out = LLM(completions)
per_token_logps = F.log_softmax(llm_out, dim=-1)  # (B*G, L)

# get reference logprobs for each action
ref_out = REF(completions)
ref_per_token_logps = F.log_softmax(ref_out, dim=-1)  # (B*G, L)

# compute KL divergence between policy and reference policy
kl_div = per_token_logps - ref_per_token_logps

# alternative KL divergence used by DeepSeekMath [1]
kl_div_alt = (
    torch.exp(ref_per_token_logps - per_token_logps)
    - (ref_per_token_logps - per_token_logps)
    - 1
)

# compute mean and std of grouped rewards
reward_mean = rewards.view(-1, G).mean(dim=1)  # (B,)
reward_std = rewards.view(-1, G).std(dim=1)  # (B,)

# compute advantage for GRPO
advantage = (rewards.view(-1, G) - reward_mean)
advantage /= (reward_std + 1e-8)  # (B, G)
advantage = advantage.view(-1, 1)  # (B*G, 1)

# compute the policy ratio
policy_ratio = torch.exp(
    per_token_logps - old_per_token_logps,
)  # (B*G, L)
clip_policy_ratio = torch.clamp(
    policy_ratio,
    min=1.0 - eps,
    max=1.0 + eps,
)

# compute clipped loss
loss = torch.min(
    advantage * policy_ratio,
    advantage * clip_policy_ratio,
)  # (B*G, L)

# kl divergence added as penalty term to loss
loss = -loss + kl_beta * kl_div

# aggregate the loss across tokens (many options exist here)
loss = ((loss * completion_mask).sum(axis=-1) /
        completion_mask.sum(axis=-1)).mean()

# perform policy gradient update
optimizer.zero_grad()
loss.backward()
optimizer.step()