import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from qlora_demo.data_utils import prepare_dataset

# Choose the device: MPS on Apple Silicon, otherwise CPU.
device = "mps" if torch.backends.mps.is_available() else "cpu"
# Base model name and output folder for adapters.
MODEL_ID = "distilgpt2"
OUTPUT_DIR = "qlora_adapter"


def build_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # !r :
    # !r controls the size (capacity) of the LoRA adapters — the higher the r,
    # !the more trainable parameters and the greater the model's ability to learn task-specific changes.
    # !Instead of learning all 16.7 million values, LoRA learns:
    # !A = 4096 × r =  4096 × 8
    # !B = r × 4096 =  8 × 4096
    # !4096×8 + 8×4096 =  65,536, instead of 16,777,216 parameters.
    # !This is a significant reduction in trainable parameters.

    # !lora_alpha is the scaling factor for the LoRA updates.
    # !Scaling controls how much influence the LoRA adapter's learned update has when it's added to the frozen model weights.
    # !Scaling = lora_alpha/r = 32/8 = 4.0, which means the LoRA updates are scaled by a factor of 4.0 before being added to the original weights.

    # !target_modules specifies which layers (weight matrices) in the model should receive LoRA adapters.
    # !For GPT2-style models, "c_attn" and "c_proj" are the attention and projection layers where LoRA is applied.
    # !Only these modules get the trainable A and B matrices. All other layers remain completely frozen.

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def train():
    model, tokenizer = build_model()
    train_dataset = prepare_dataset("dataset/train.jsonl", tokenizer)

    training_args = TrainingArguments(
        output_dir="./training_output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        logging_steps=20,
        save_strategy="no",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved adapter and tokenizer to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
