# Low-Rank & Sparse Transcoders for Model Stitching (LLMs)

This repo trains **low-rank + sparse transcoders** to map hidden activations from Model **A** to Model **B** at a chosen layer, then evaluates:
- **Cross-reconstruction** (A→B MSE, CKA, SVCCA).
- **Stitched next-token loss**: replace B's hidden activations with mapped A activations during forward pass.
- **Causal transfer**: translate a direction from A and inject into B; compare next-token distributions.

Baselines included:
- **Linear** mapping (full matrix).
- **Low-rank** mapping (LoRA-style `W = U V^T` with optional residual).
- **Low-rank + L1 sparsity** (proximal soft-threshold on U and V).
- **Tiny MLP** baseline (2-layer GELU) with comparable parameter budget.

Uses **TransformerLens** for stable hooks. Swap models in `configs/default.yaml`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Collect paired activations A & B for training/eval (choose layer(s))
python scripts/collect_acts.py --config configs/default.yaml --layers 6

# 2) Train transcoders & baselines
python scripts/train_transcoder.py --config configs/default.yaml --layer 6 --model lowrank --rank 128 --l1 1e-4
python scripts/train_transcoder.py --config configs/default.yaml --layer 6 --model linear
python scripts/train_transcoder.py --config configs/default.yaml --layer 6 --model mlp --hidden 512

# 3) Evaluate mapping quality + stitched next-token loss
python scripts/eval_stitching.py --config configs/default.yaml --layer 6 --checkpoint runs/demo/lowrank_L6.pt

# 4) Causal transfer sanity check
python scripts/causal_transfer.py --config configs/default.yaml --layer 6   --checkpoint runs/demo/lowrank_L6.pt --text "The capital of France is"
```

Outputs: `runs/<run_name>/...` (checkpoints, CSVs, logs).

