Quick guide: Run LoRA demo on macOS (M4) in VS Code

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

3) Train the adapter using the package module:

```bash
python3 -m qlora_demo.train
```

4) Evaluate the saved adapter on held-out data:

```bash
python3 -m qlora_demo.eval
```

5) Run inference after training:

```bash
python3 -m qlora_demo.inference
```

Notes:
- This demo uses `distilgpt2` and PEFT LoRA; it does not use bitsandbytes or QLoRA 4-bit quantization because those require NVIDIA CUDA GPUs.
- If you want a Colab notebook for true QLoRA (bitsandbytes + GPU), ask and I will prepare it.
