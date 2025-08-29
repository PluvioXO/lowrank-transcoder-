\
import os, random
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_texts(dataset: str, subset: Optional[str], split: str, text_column: str, n: int) -> List[str]:
    if dataset == "wikitext":
        ds = load_dataset("wikitext", subset or "wikitext-103-raw-v1", split=split)
    else:
        ds = load_dataset(dataset, subset, split=split)
    out = []
    for ex in ds:
        t = ex.get(text_column, None)
        if t and t.strip():
            out.append(t)
            if len(out) >= n: break
    return out

def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch; batch = []
    if batch: yield batch
