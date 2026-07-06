import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Choose the device: MPS on Apple Silicon, otherwise CPU.
device = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "distilgpt2"
OUTPUT_DIR = "lora_adapter"


def build_model():
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    model.eval()
    return model, tokenizer


def generate_samples(model, tokenizer):
    prompts = [
        "User: Greet a new friend politely.",
        "User: Translate 'hello' to Spanish.",
        "User: What is 7 plus 8?",
        "User: Explain how to make tea.",
    ]

    print("\nGenerated outputs:")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )
        print("\nPrompt:", prompt)
        print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    model, tokenizer = build_model()
    generate_samples(model, tokenizer)
