from typing import Callable, Dict, List, Literal, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.baseline import evaluate_vllm

# ========== SFT Part in the assignment ==========

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str], 
    tokenizer: PreTrainedTokenizer,
    device: str = "mps"
) -> Dict[str, torch.Tensor]:
    B = len(prompt_strs)

    pad_id = tokenizer.pad_token_id

    prompts_tok = tokenizer(prompt_strs).input_ids
    output_tok = tokenizer(output_strs).input_ids
    
    combined = []
    for p,o in zip(prompts_tok, output_tok):
        combined.append(p + o)
    
    prompt_and_output_lens = max(len(x) for x in combined)
    
    inputs_id_full = torch.full(
        size=(B, prompt_and_output_lens),
        fill_value=pad_id,
    )
    response_mask = torch.zeros_like(inputs_id_full)
    
    for i, (full_seq, prompt_tok, output_tok) in enumerate(zip(combined, prompts_tok, output_tok)):
        seq_len = len(full_seq)
        prompt_len = len(prompt_tok)
        output_len = len(output_tok)

        inputs_id_full[i, :seq_len] = torch.tensor(full_seq)
        response_mask[i, prompt_len:prompt_len+output_len] = 1
    
    input_ids = inputs_id_full[:, :-1].to(device)
    labels = inputs_id_full[:, 1:].clone().to(device)
    response_mask = response_mask[:, 1:].bool().to(device)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # turn the logits to log-prob
    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim = True)
    # calculate the entropy
    entropy_loss = -torch.sum(torch.exp(log_prob) * log_prob, dim=-1)
    return entropy_loss

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    device: str = "mps"
) -> Dict[str, torch.Tensor]:
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    model = model.to(device)
    input_ids = input_ids.to(device)
    labels = labels.to(device, dtype=torch.long)
    
    model.eval()
    logits = model(input_ids).logits # This return (batch_size, seq_len, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)
    wanted_labels = labels.unsqueeze(-1)
    chosen_log_probs = log_probs.gather(dim=-1, index=wanted_labels).squeeze(-1)

    result = {
        "log_probs": chosen_log_probs
    }

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None= None,
) -> torch.Tensor:
    return torch.sum(tensor * mask, dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ce = -policy_log_probs
    ce_normalized = masked_normalize(
        tensor = ce,
        mask = response_mask,
        normalize_constant = normalize_constant, 
        dim = -1
    )
    loss = ce_normalized.mean()
    scaled_loss = loss / gradient_accumulation_steps
    
    meta_data = {
        "loss": loss,
        "scaled_loss": scaled_loss,
        "response_lengths": response_mask.sum(-1)
    }
    
    return scaled_loss, meta_data

def log_generations(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        answers: List[str],
        sampling_params: SamplingParams,
        log_file: str = None,
        iter_idx: int = 0
):
    """
    Log the generations of the model.
    """
    results = evaluate_vllm(vllm_model, reward_fn, prompts, answers, sampling_params, save_path = None)

    with open(log_file, 'a') as f:
        f.write("-" * 100 + "\n")
        f.write(f"ITERATION {iter_idx}\n")
        for result in results:
            f.write(f"\nprompt: {result['prompt']}\nresponse: {result['model_output']}\nanswer: {result['expected_answer']}\nformat_reward: {result['format_reward']}\nanswer_reward: {result['answer_reward']}\nreward: {result['reward']}\n\n")
        
        f.write("-" * 100 + "\n")
        
# ========== GRPO Part in the assignment ==========

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            f"rollout_responses and repeated_ground_truths must have same length, "
            f"got {len(rollout_responses)} and {len(repeated_ground_truths)}"
        )
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    
    rollout_batch_size = len(rollout_responses)
    n_groups = rollout_batch_size // group_size
    
    raw_rewards_list = []
    format_rewards_list = []
    answer_rewards_list = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(resp, gt)
        raw_rewards_list.append(scores["reward"])
        if "format_reward" in scores:
            format_rewards_list.append(float(scores["format_reward"]))
        if "answer_reward" in scores:
            answer_rewards_list.append(float(scores["answer_reward"]))
    
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    
    grouped = raw_rewards.view(n_groups, group_size)
    
    group_means = grouped.mean(dim=1, keepdim=True)
    centered = grouped - group_means
    
    if normalize_by_std:
        group_stds = grouped.std(dim=1, keepdim=True, unbiased=True)
        denom = group_stds + advantage_eps
        normalized = centered / denom
    else:
        normalized = centered
    
    advantages = normalized.reshape(-1)

    metadata: dict[str, float] = {
        "reward/mean": raw_rewards.mean().item(),
        "reward/std": raw_rewards.std(unbiased=False).item(),
        "reward/min": raw_rewards.min().item(),
        "reward/max": raw_rewards.max().item(),
        "advantage/mean": advantages.mean().item(),
        "advantage/std": advantages.std(unbiased=False).item(),
    }

    if format_rewards_list:
        fr = torch.tensor(format_rewards_list, dtype=torch.float32)
        metadata["format_reward/mean"] = fr.mean().item()
    if answer_rewards_list:
        ar = torch.tensor(answer_rewards_list, dtype=torch.float32)
        metadata["answer_reward/mean"] = ar.mean().item()

    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    # −At·log pθ (ot|q, o<t)
    return -1 * (raw_rewards_or_advantages * policy_log_probs)

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # K = (πθ (ot|q, o<t)/πθold (ot|q, o<t)
    # −min (K * At , clip(K, 1−ϵ, 1 + ϵ) * At)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clip = torch.clip(ratio, 1 - cliprange, 1 + cliprange)
    
    unclipped_obj = ratio * advantages
    clipped_obj = clip * advantages
    
    is_clipped = (clipped_obj < unclipped_obj)

    loss = -1 * torch.min(unclipped_obj, clipped_obj)
    metadata = {
        "is_clipped": is_clipped
    }
    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    allowed_loss_types = [
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip"
    ]
    assert (
        loss_type in allowed_loss_types, 
        f"`loss_type`must be one of the {", ".join(allowed_loss_types)}, got {loss_type}"
    )
    
    if loss_type == "no_baseline" and raw_rewards is None:
        raise ValueError('`raw_rewards`is required when using `loss_type="no_baseline"`')

    if loss_type in [
        "reinforce_with_baseline",
        "grpo_clip"
    ] and advantages is None:
        raise ValueError(
            '`advantages`is required when using `loss_type="reinforce_with_baseline"` or `loss_type="grpo_clip"`'
        )
    
    if loss_type == "grpo_clip" and (old_log_probs is None or cliprange is None):
        raise ValueError(
            '`old_log_probs` and `cliprange` is required when using `loss_type="grpo_clip"`'
        )
    
    if loss_type == "grpo_clip":
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )
    
    elif loss_type == "no_baseline":
        rewards_or_advantages = raw_rewards
    else:
        rewards_or_advantages = advantages

    metadata =  {
        "reward_mean": rewards_or_advantages.mean(),
        "reward_std": rewards_or_advantages.std()
    }
    return compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages=rewards_or_advantages,
        policy_log_probs=policy_log_probs
    ), metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
) -> torch.Tensor:
    return torch.sum(tensor * mask, dim=dim) / mask.sum(dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_per_token, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )

    masked_mean_pg_loss = masked_mean(
        loss_per_token,
        response_mask,
        -1
    )
    pg_loss = masked_mean_pg_loss.mean()
    scaled_pg_loss = pg_loss / gradient_accumulation_steps

    metadata.update({
        "pg_loss": pg_loss ,
        "scaled_pg_loss": scaled_pg_loss,
        "response_lengths": response_mask.sum(-1)
    })

    scaled_pg_loss.backward()
    return scaled_pg_loss, metadata