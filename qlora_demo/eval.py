import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from peft import PeftModel
from qlora_demo.data_utils import prepare_dataset

# Device selection: MPS if available, otherwise CPU.
device = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "distilgpt2"
OUTPUT_DIR = "qlora_adapter"
EVAL_DATA_PATH = "dataset/eval.jsonl"
BATCH_SIZE = 4


def build_model():
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    model.to(device)
    model.eval()
    return model, tokenizer


def compute_eval_loss(model, eval_dataset):
    data_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE)
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            batch_count += 1

    avg_loss = total_loss / batch_count if batch_count else float("nan")
    return avg_loss


def generate_samples(model, tokenizer):
    prompts = [
        "User: Explain how to make tea.",
        "User: Translate 'hello' to Spanish.",
        "User: What is 7 plus 8?",
        "User: Give a helpful tip for studying.",
    ]

    print("\nSample generations:")
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
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nPrompt:", prompt)
        print("Output:", text)


def main():
    print(f"Loading evaluation dataset from {EVAL_DATA_PATH}")
    model, tokenizer = build_model()
    eval_dataset = prepare_dataset(EVAL_DATA_PATH, tokenizer)
    loss = compute_eval_loss(model, eval_dataset)
    perplexity = torch.exp(torch.tensor(loss)).item()
    print(f"\nEvaluation loss: {loss:.4f}")
    print(f"Evaluation perplexity: {perplexity:.2f}")
    generate_samples(model, tokenizer)


if __name__ == "__main__":
    main()
