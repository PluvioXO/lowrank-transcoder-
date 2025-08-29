\
from typing import List
import torch
from dataclasses import dataclass
from transformer_lens import HookedTransformer

@dataclass
class ModelHandle:
    name: str
    model: HookedTransformer
    tokenizer: any

def load_model(name: str) -> ModelHandle:
    m = HookedTransformer.from_pretrained(name, device="cuda" if torch.cuda.is_available() else "cpu")
    return ModelHandle(name=name, model=m, tokenizer=m.tokenizer)

def hook_name(layer: int, kind: str) -> str:
    mp = {
        "resid_pre": f"blocks.{layer}.hook_resid_pre",
        "mlp_out": f"blocks.{layer}.mlp.hook_post",
        "attn_out": f"blocks.{layer}.hook_attn_out",
    }
    return mp[kind]

@torch.no_grad()
def get_layer_acts(h: ModelHandle, texts: List[str], layer: int, kind: str, max_length: int):
    """Concatenate interior-token activations across texts: [N_tokens, d_model]."""
    hn = hook_name(layer, kind)
    X = []
    for t in texts:
        toks = h.tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length).to(h.model.cfg.device)
        _, cache = h.model.run_with_cache(toks, names_filter=[hn])
        A = cache[hn][0]  # [seq, d_model] (batch dim collapsed via [0])
        if A.shape[0] > 2:
            X.append(A[1:-1].detach().cpu())  # avoid BOS/EOS
    if not X:
        raise RuntimeError("No activations captured.")
    return torch.cat(X, dim=0)

def stitch_with_mapping(Ah: ModelHandle, Bh: ModelHandle, layer: int, kind: str, mapper_fn, texts: List[str], max_length: int, max_batches: int=50):
    """
    Stitched evaluation: replace B's activation at (layer,kind) with mapped A activations for the same text.
    mapper_fn: function that maps A's [seq, d_in] → [seq, d_out] on CPU tensor.
    Returns mean token-level cross-entropy (approx NLL).
    """
    import torch.nn.functional as F
    hn = hook_name(layer, kind)
    nlls = []
    for i, t in enumerate(texts[:max_batches]):
        toksA = Ah.tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length).to(Ah.model.cfg.device)
        toksB = Bh.tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length).to(Bh.model.cfg.device)
        _, cacheA = Ah.model.run_with_cache(toksA, names_filter=[hn])
        Aact = cacheA[hn][0].detach()  # [seq, d_in]
        mapped = mapper_fn(Aact).to(Bh.model.cfg.device)  # [seq, d_out]

        def hook_fn(act, hook):
            # act shape: [batch, seq, d_out] or [seq, d_out]
            if act.ndim == 3:
                if mapped.ndim == 2:
                    act[:, :, :] = mapped.unsqueeze(0)
                else:
                    act[:] = mapped
            else:
                act[:] = mapped
            return act

        with Bh.model.hook_points[hn].register_hook(hook_fn):
            logits = Bh.model(toksB).logits  # [1, seq, vocab]
        shift_logits = logits[:, :-1, :]
        shift_labels = toksB.input_ids[:, 1:]
        loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), reduction="mean")
        nlls.append(loss.item())
    return sum(nlls)/len(nlls)
