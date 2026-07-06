from typing import Dict
from transformers import AutoTokenizer
from datasets import DatasetDict, load_dataset


def prepare_dataset(path: str, tokenizer: AutoTokenizer, max_length: int = 128):
    """Load and tokenize a JSONL dataset for causal language modeling."""
    dataset = load_dataset("json", data_files={"data": path}, split="data")

    def tokenize_fn(examples: Dict):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        labels = []
        for input_ids in tokenized["input_ids"]:
            label_ids = [tok if tok != tokenizer.pad_token_id else -100 for tok in input_ids]
            labels.append(label_ids)
        tokenized["labels"] = labels
        return tokenized

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type="torch")
    return dataset
