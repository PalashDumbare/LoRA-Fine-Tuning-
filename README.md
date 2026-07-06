Quick guide: Run LoRA demo in VS Code

1) Create and activate a Python environment (recommended: venv or conda)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2) Install required packages. For macOS M1/M2/M4 use the PyTorch macOS instructions from https://pytorch.org/get-started/locally/ — example CPU/MPS install:

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets peft accelerate
```

3) Optionally use the unified launcher:

```bash
python3 run.py
```

Then choose one of:
- `train`
- `eval`
- `infer`

4) Train the adapter using the package module:

```bash
python3 -m lora_demo.train
```

5) Evaluate the saved adapter on held-out data:

```bash
python3 -m lora_demo.eval
```

6) Run inference after training:

```bash
python3 -m lora_demo.inference
```


