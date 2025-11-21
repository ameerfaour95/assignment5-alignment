from cs336_alignment import utils
from drgrpo_grader import r1_zero_reward_fn
import json
import torch
import wandb
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import Optional, Dict, List
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
from unittest.mock import patch

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
        model=model_id,
        device=device,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def load_training_data(path: str):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return (
        [d["prompt"] for d in data],
        [d["response"] for d in data],
        [d["ground_truth"] for d in data]
    )

def load_SFT(path: str):
    t_p, t_res, t_gt = load_training_data(path=path)
    train_idxs = random.sample(range(len(t_p)), len(t_p))
    train_prompts = [t_p[i] for i in train_idxs]
    train_responses = [t_res[i] for i in train_idxs]
    train_ground_truths = [t_gt[i] for i in train_idxs]

    return (
        train_prompts,
        train_responses,
        train_ground_truths
    )

def training_step(
    tokenized_batch: Dict[str, torch.Tensor],
    model: PreTrainedModel,
    gradient_accumulation_steps: int,
    return_token_entropy: bool,
    device: str,
):
    log_probs_and_token_entropy_dict = utils.get_response_log_probs(
        model=model,
        input_ids=tokenized_batch['input_ids'],
        labels=tokenized_batch['labels'],
        return_token_entropy=return_token_entropy,
        device=device
    )

    log_probs = log_probs_and_token_entropy_dict['log_probs']
    token_entropy = log_probs_and_token_entropy_dict['token_entropy']
    token_entropy *= tokenized_batch['response_mask']
    token_entropy /= torch.sum(tokenized_batch['response_mask'], dim=-1, keepdim=True)
    avg_over_responses = token_entropy.mean()

    scaled_loss, meta_data = utils.sft_microbatch_train_step(
        policy_log_probs=log_probs,
        response_mask=tokenized_batch['response_mask'],
        gradient_accumulation_steps=gradient_accumulation_steps,
        normalize_constant=1.0
    )
    
    scaled_loss.backward()
    
    return scaled_loss.item(), avg_over_responses.item()

def wandb_setup():
    wandb.define_metric("train_step")  # x-axis for training and eval
    wandb.define_metric("iter")        # x-axis for iteration-level metrics

    # Both train/ and eval/ metrics use train_step as x-axis
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="train_step")  # Note: uses train_step, not eval_step

    # iter/ metrics use iter as x-axis
    wandb.define_metric("iter/*", step_metric="iter")

def train_sft(
    train_prompts: List[str],
    train_responses: List[str],
    train_ground_truths: List[str],
    val_prompts: Optional[List[str]],
    val_answers: Optional[List[str]],
    vllm_model: Optional[LLM],
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    tokenizer: PreTrainedTokenizer,
    config: dict,
    start_training_step: int,
    eval_sampling_params: Optional[SamplingParams],
):
    do_eval = vllm_model is not None
    if do_eval:
        assert val_prompts is not None and val_answers is not None, (
            "If you want to preform eval you have to pass `val_prompts` and `val_answers`"
        )
    
    minibatch_size = config["minibatch_size"]
    batch_size = config['train_batch_size']
    device = config["device"]

    train_step = start_training_step
    
    learning_rate = config['learning_rate']
    log_every_n = config['log_every_n']
    eval_every_n = config['eval_every_n']
    device = config["device"]
    print("Running SFT training...")
    assert config["train_batch_size"] % config["minibatch_size"] == 0, \
        "train_batch_size must be an integer multiple of minibatch_size"
    gradient_accumulation_steps = config["train_batch_size"] // config["minibatch_size"]

    print(f"Training for {config['n_epochs']} epochs...")
    mini_train_step = 0
    log_train = False
    log_eval = False
    for epoch in range(config['n_epochs']):
        print(f"Running epoch {epoch}...")
        
        train_indices = list(range(len(train_prompts)))
        random.shuffle(train_indices)
        train_prompts = [train_prompts[i] for i in train_indices]
        train_responses = [train_responses[i] for i in train_indices]
        train_ground_truths = [train_ground_truths[i] for i in train_indices]
        
        for idx in range(0, len(train_prompts), minibatch_size):
            minibatch_prompts = train_prompts[idx: idx+minibatch_size]
            minibatch_responses = train_responses[idx: idx+minibatch_size]
            
            tokenized_minibatch = utils.tokenize_prompt_and_output(
                prompt_strs=minibatch_prompts,
                output_strs=minibatch_responses,
                tokenizer=tokenizer,
                device=device
            )
            loss_val, avg_entropy = training_step(
                tokenized_batch=tokenized_minibatch,
                model=model,
                gradient_accumulation_steps=gradient_accumulation_steps,
                return_token_entropy=True,
                device=device
            )
            if (mini_train_step + 1) % gradient_accumulation_steps == 0:
                # gradient cliping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                train_step += 1
                
                if train_step % log_every_n == 0:
                    log_train = True
                if train_step % eval_every_n == 0 and do_eval:
                    log_eval = True
                
                if log_train:
                    wandb.log(
                        {
                            "train/loss": loss_val,
                            "train/avg_entropy": avg_entropy,
                            "train_step": train_step
                        }
                    )
                    log_train = False
                
                if log_eval:
                    print("Training step: ", train_step)
                    
                    print("Loading policy into vllm...")
                    model.eval()
                    with torch.no_grad():
                        load_policy_into_vllm_instance(model, vllm_model)
                    model.train()
                    
                    small_k = max(1, min(minibatch_size, int(len(val_prompts) * 0.2)))
                    log_prompts_indices = random.sample(range(len(val_prompts)), small_k)
                    val_minibatch_prompts = [val_prompts[i] for i in log_prompts_indices]
                    val_minibatch_answers = [val_answers[i] for i in log_prompts_indices]
                    print(f"Logging generations for {len(log_prompts_indices)} prompts...")
                    utils.log_generations(
                        vllm_model=vllm_model,
                        reward_fn=r1_zero_reward_fn,
                        prompts=val_minibatch_prompts,
                        answers=val_minibatch_answers,
                        sampling_params=eval_sampling_params,
                        log_file = f'sft_results/sft_{batch_size}_{minibatch_size}_{learning_rate}.txt'
                    )
                    log_eval = False

            mini_train_step += 1

    partial_step = (mini_train_step % gradient_accumulation_steps)
    print(f"Partial gradient accumulation steps: {partial_step}")
    if partial_step > (gradient_accumulation_steps // 2):
        print("Performing partial gradient update...")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        train_step += 1
    else:
        print("Not performing gradient update...")
        # erase gradients
        optimizer.zero_grad()

    print("Training complete!")
    if log_eval:
        model.eval()
        with torch.no_grad():
            load_policy_into_vllm_instance(model, vllm_model)
        model.train()
    
    #TODO: Preform last full eval

def load_models(
    model_id: str,
    device: str,
    seed: int,
    use_flash_attention_2: bool,
    use_vllm_for_eval: bool,
):
    
    if use_flash_attention_2:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if use_vllm_for_eval:
        assert device.startswith("cuda"), "vLLM currently requires CUDA"
        print("Initializing vllm...")
        vllm_model = init_vllm(
            model_id,
            device = device,
            seed = seed,
            gpu_memory_utilization = 0.85
        )

        print("Loading policy into vllm...")
        load_policy_into_vllm_instance(model, vllm_model)
    else:
        vllm_model = None
    
    return model, vllm_model, tokenizer

def train_start(
    config: dict,
    use_flash_attention_2: bool,
    use_vllm_for_eval: bool,
    eval_sampling_params: Optional[SamplingParams],
):
    if use_flash_attention_2:
        assert config["device"] != "mps", (
            "flash attention doesnt support `mps` devices"
        )
        assert torch.cuda.is_available(), (
            "flash attention 2 is only available for cuda"
        )

    model, vllm_model, tokenizer = load_models(
        model_id = config['model'],
        device = config['device'],
        seed = config['seed'],
        use_flash_attention_2 = use_flash_attention_2,
        use_vllm_for_eval = use_vllm_for_eval
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = config['learning_rate'])
    
    print("Loading training sets...")
    train_prompts, train_responses, train_ground_truths = load_SFT(config["train_path"])
    print(f"Loaded {len(train_prompts)} training examples")
    
    val_prompts, val_answers = None, None
    if config['val_path'] is not None:
        print("Loading validation sets...")
        val_prompts, val_answers, _ = load_SFT(config["val_path"])
        print(f"Loaded {len(val_prompts)} validation examples")
    
    train_sft(
        train_prompts,
        train_responses,
        train_ground_truths,
        val_prompts,
        val_answers,
        vllm_model,
        model,
        optimizer,
        tokenizer,
        config,
        start_training_step,
        eval_sampling_params
    )
    
    model.save_pretrained(config['output_model_dir'])
    tokenizer.save_pretrained(config['output_model_dir'])

if __name__ == "__main__":
    val_path = None
    train_path = "/Users/ameefaour/Desktop/CS336_LLM_from_scratch/assignment5-alignment/data/filtered_train.jsonl"
    val_path = "/Users/ameefaour/Desktop/CS336_LLM_from_scratch/assignment5-alignment/data/filtered_validation.jsonl"
    
    config = {
        "model": "Qwen/Qwen2.5-Math-1.5B",
        "minibatch_size": 8,
        "n_steps": 64,
        "train_batch_size": 128,
        "learning_rate": 1e-4,
        "seed": 42,
        "train_path": train_path,
        "val_path": val_path,
        "device": "mps",
        "log_every_n": 16,
        "eval_every_n": 32,
    }
        
    start_training_step = 0
    dataset_len = sum(1 for _ in open(train_path))
    # ceil
    steps_per_epoch = max(1, (dataset_len + config["train_batch_size"] - 1) // config["train_batch_size"])
    config['n_epochs'] = max(1, config["n_steps"] // steps_per_epoch)

    models_dir = "/Users/ameefaour/Desktop/CS336_LLM_from_scratch/assignment5-alignment/cs336_alignment/models/"
    output_model_dir = f"filterd_sft_{config['train_batch_size']}_{config['learning_rate']}"
    config['output_model_dir'] = models_dir + output_model_dir

    wandb.init(project = "cs336-alignment-sft", 
                name = f"filtered_sft_{config['train_batch_size']}_{config['learning_rate']}", 
                config = config)
    wandb_setup()

    # eval_sampling_params = SamplingParams(
    #     temperature = 1.0, 
    #     top_p = 1.0, 
    #     max_tokens = 1024, 
    #     stop = ["</answer>"], 
    #     include_stop_str_in_output = True,
    # )
    
    set_seed(config["seed"])

    use_flash_attention_2 = (config["device"].startswith("cuda") and torch.cuda.is_available())

    train_start(
        config,
        use_flash_attention_2=use_flash_attention_2,
        use_vllm_for_eval=False,
        eval_sampling_params=None,
    )
