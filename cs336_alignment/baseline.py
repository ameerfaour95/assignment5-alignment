import argparse
import json
from vllm import LLM, SamplingParams
from typing import Callable, List, Dict
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import re

def prepare_prompts_format(path: str, prompt_path: str) -> List[dict]:
    with open(prompt_path, "r") as f:
        base_prompt = f.read().strip()
    prompts = []
    with open(path, "r") as json_file:
        for line in json_file:
            ex = json.loads(line)
            assert "question" in ex and "answer" in ex, "`question` and `answer` must exist."
            question = ex["question"]
            # Gold is after '####'
            m = re.search(r"(?s)####\s*(.*?)\s*$", ex["answer"])
            gold = m.group(1).strip() if m else ex["answer"].strip()
            prompt = base_prompt.replace("{question}", str(question))
            prompts.append({"question": question, "prompt": prompt, "true_answer": gold})
    return prompts

def parse_result(output: str):
    match = re.search(r"(?s)<answer>(.*?)</answer>", output)
    if match:
        answer = match.group(1).strip()
    else:
        answer = ""
    return answer

def reward_fn(
    output: str,
    reference: str
):
    return {
        "accuracy": float(output.strip() == reference.strip())
    }

def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        answers: List[str],
        eval_sampling_params: SamplingParams,
        save_path: str = None,
        ) -> float:
    """
    Evaluate the VLLM model using the reward function.
    """
    # serialization
    results = []

    # generate text
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # grade outputs
    for prompt, output, answer in zip(prompts, outputs, answers):
        # get output response
        response = output.outputs[0].text
        scores = reward_fn(response, answer)
        results.append({
            "prompt": prompt,
            "model_output": response,
            "expected_answer": answer,
            "format_reward": scores["format_reward"],
            "answer_reward": scores["answer_reward"],
            "reward": scores["reward"]
        })
    
    if save_path is not None:
        with open(save_path, 'a') as f:
            # save json list of dicts
            json.dump(results, f, indent = 2)
    
    return results

def run_math_baseline(model_path: str, json_path: str, temperature: float, top_p: float, max_tokens: int):
    examples = prepare_prompts_format(json_path, "cs336_alignment/prompts/r1_zero.prompt")
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        stop=["</answer>"], include_stop_str_in_output=True
    )
    llm = LLM(model=model_path)
    evaluate_vllm(llm, r1_zero_reward_fn, examples, sampling_params)
        
def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5 Math model on MATH dataset")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Name or path of the model (e.g., Qwen2.5-Math-1.5B)")
    parser.add_argument("--data_path", type=str, default="/data/a5-alignment/MATH/validation.jsonl",
                        help="Path to MATH validation data")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)

    args = parser.parse_args()
    
    run_math_baseline(
        args.model_path,
        args.data_path,
        args.temperature,
        args.top_p,
        args.max_tokens
    )

if __name__=="__main__":
    main()