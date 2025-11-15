from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.baseline import evaluate_vllm


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
        size=(B,prompt_and_output_lens),
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
    with torch.no_grad():
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