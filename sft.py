import torch
import random
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name: str, lora_config: LoraConfig):
    """Load tokenizer and model, wrap model with PEFT LoRA.

    Returns (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir="./model_cache")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto",cache_dir="./model_cache"
    )
    model = get_peft_model(model, lora_config)
    return tokenizer, model

def train(model, dataset, training_args, tokenizer):
    """Run SFT training loop.

    model: PEFT-wrapped model
    dataset: a HuggingFace Dataset or dataset dict
    training_args: SFTConfig
    tokenizer: tokenizer
    """
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

if __name__ == "__main__":

    model_name = "Qwen/Qwen2.5-1.5B"

    lora_config = LoraConfig(
        r=16,  # rank — higher = more capacity, more memory
        lora_alpha=32,  # scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

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

    sft_dataset = load_from_disk('./sft_dataset')

    tokenizer, model = load_model(model_name, lora_config)
    train(model, sft_dataset, training_args, tokenizer)












