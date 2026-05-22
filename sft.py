import re
import torch
import random
from datasets import load_dataset
from sympy import simplify, sympify
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from sympy.parsing.latex import parse_latex
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_boxed_answer(solution: str) -> str | None:
    # Step 1: find where \boxed{ starts
    match = re.search(r"\\boxed\{", solution)
    if not match:
        return None

    # Step 2: manually walk forward counting braces
    # to handle nested curly braces like \frac{}{}
    start = match.end()  # position right after \boxed{
    depth = 1
    i = start

    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1

    return solution[start : i - 1]  # content between the outer braces


def answers_match(predicted: str, ground_truth: str) -> bool:
    # First try exact string match after stripping whitespace
    if predicted.strip() == ground_truth.strip():
        return True

    # Then try symbolic math comparison
    try:
        pred_expr = parse_latex(predicted)
        gt_expr = parse_latex(ground_truth)
        return simplify(pred_expr - gt_expr) == 0
    except Exception:
        pass

    # Finally try float comparison
    try:
        pred_float = float(sympify(predicted))
        gt_float = float(sympify(ground_truth))
        return abs(pred_float - gt_float) < 1e-6
    except Exception:
        return False


def is_verifiable(example) -> bool:
    answer = extract_boxed_answer(example["solution"])
    return answer is not None


def format_as_messages(example):
    answer = extract_boxed_answer(example["solution"])
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
            {
                "role": "assistant",
                "content": f"<think>\n{example['solution']}\n</think>\n\nThe answer is $\\boxed{{{answer}}}$",
            },
        ]
    }


def format_for_sft(example) -> dict:
    answer = extract_boxed_answer(example["solution"])

    return {
        "prompt": SYSTEM_PROMPT + "\n\nProblem: " + example["problem"],
        "response": f"<think>\n{example['solution']}\n</think>\n\nThe answer is $\\boxed{{{answer}}}$",
        "answer": answer,
    }


model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")

# Filter dataset
filtered = dataset.filter(is_verifiable, num_proc=4)


SYSTEM_PROMPT = """You are a mathematical reasoning assistant. 
For each problem, think through it step by step inside <think></think> tags, 
then give your final answer in \\boxed{}."""

# Apply formatting
formatted = filtered.map(format_for_sft, num_proc=4)

# Take 100K subset for SFT — no need for all 836K
sft_dataset = formatted.shuffle(seed=42).select(range(100_000))

# Save it
sft_dataset.save_to_disk("sft_dataset")

lora_config = LoraConfig(
    r=16,  # rank — higher = more capacity, more memory
    lora_alpha=32,  # scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)


sft_dataset = filtered.shuffle(seed=42).select(range(100_000))
sft_dataset = sft_dataset.map(format_as_messages, num_proc=4)

training_args = SFTConfig(
    output_dir="sft_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=50,
    save_steps=500,
    fp16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=sft_dataset,
)

# trainer.train()
