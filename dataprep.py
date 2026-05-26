import re
from typing import Optional, Dict, Any

from datasets import load_dataset


SYSTEM_PROMPT = """You are a mathematical reasoning assistant. 
For each problem, think through it step by step inside <think></think> tags, 
then give your final answer in \\boxed{}."""


def extract_boxed_answer(solution: str) -> str | None:
    # Step 1: find where \boxed{ starts
    match = re.search(r'\\boxed\{', solution)
    if not match:
        return None
    # Step 2: manually walk forward counting braces
    # to handle nested curly braces like \frac{}{}
    start = match.end()  # position right after \boxed{
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == '{':
            depth += 1
        elif solution[i] == '}':
            depth -= 1
        i += 1
    return solution[start:i-1]  # content between the outer braces


def is_verifiable(example: Dict[str, Any]) -> bool:
    """Return True if the example has a verifiable boxed answer."""
    answer = extract_boxed_answer(example.get("solution", ""))
    return answer is not None

def format_as_messages(example: Dict[str, Any]) -> Dict[str, Any]:
    answer = extract_boxed_answer(example.get("solution", ""))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example.get("problem", "")},
            {
                "role": "assistant",
                "content": f"<think>\n{example.get('solution', '')}\n</think>\n\nThe answer is $\\boxed{{{answer}}}$",
            },
        ]
    }

def format_for_sft(example: Dict[str, Any]) -> Dict[str, Any]:
    answer = extract_boxed_answer(example.get("solution", ""))

    return {
        "prompt": SYSTEM_PROMPT + "\n\nProblem: " + example.get("problem", ""),
        "completion": f"<think>\n{example.get('solution', '')}\n</think>\n\nThe answer is $\\boxed{{{answer}}}$",
        "answer": answer,
    }


if __name__ == "__main__":
    
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    # Filter dataset
    filtered = dataset.filter(is_verifiable, num_proc=4)
    formatted = filtered.map(format_for_sft, num_proc=4)

    # Take 100K subset for SFT — no need for all 836K
    sft_dataset = formatted.shuffle(seed=42).select(range(100_000))

    # Save it
    sft_dataset.save_to_disk("sft_dataset")