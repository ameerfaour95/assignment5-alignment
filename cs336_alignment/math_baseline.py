import argparse
import json
from vllm import LLM, SamplingParams
from typing import Callable, List, Dict
from tqdm import tqdm
import re

PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

def prepare_prompts_format(path: str) -> List[dict]:
    with open(path, "r") as json_file:
        json_list = list(json_file)

    assert all(
        "question" in eval(f).keys() 
        and "answer" in eval(f).keys()
        for f in json_list), \
            "`question`and ànswer` must be keys in the data."
    
    prompts = []
    for q in json_list:
        valid_q = eval(q)
        question = valid_q['question']
        answer_with_explanation = valid_q['answer']
        match = re.search(r'(?s)####\s*(.*?)\s*$', answer_with_explanation)
        if match:
            answer = match.group(1).strip()
        prompt = PROMPT.replace("{question}", str(question))
        temp_dict = {
            "question": question,
            "prompt": prompt,
            "true_answer": answer
        }
        prompts.append(temp_dict)
    
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
    eval_sampling_params: SamplingParams
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    results = []
    for prompt_dict in tqdm(prompts, desc="Evaluating model"):
        prompt = prompt_dict.get("prompt")
        answer =  prompt_dict.get("true_answer")
        example =  prompt_dict.get("question")
        
        output = vllm_model.generate(prompt, eval_sampling_params)
        parsed_output = parse_result(output)
        reward = reward_fn(parsed_output, answer)
        results.append({
            "example": example,
            "true_answer": answer,
            "model_output": output,
            "score": reward
        })
    with open("math_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

def run_math_baseline(
    model_path: str,
    json_path: str,
    temperature: float,
    top_p: float,
    max_tokens: int
):
    prompts_dict = prepare_prompts_format(json_path)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    vllm_model = LLM(model=model_path)
    evaluate_vllm(
        vllm_model,
        reward_fn,
        prompts_dict,
        eval_sampling_params=sampling_params
    )
        
def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5 Math model on MATH dataset")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Name or path of the model (e.g., Qwen2.5-Math-1.5B)")
    parser.add_argument("--data_path", type=str, default="/data/a5-alignment/MATH/validation.jsonl",
                        help="Path to MATH validation data")

    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for generation")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p nucleus sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum tokens to generate per prompt")
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