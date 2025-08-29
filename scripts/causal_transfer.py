\
import os, argparse, yaml, torch
from src.backend_tlens import load_model, hook_name
from src.transcoder import TranscoderConfig, LowRankMap, LinearMap, TinyMLP

def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu")
    meta = ck["meta"]; sd = ck["state_dict"]
    if meta["model"] == "lowrank":
        U = sd["U"]; V = sd["V"]
        cfg = TranscoderConfig(d_in=V.shape[0], d_out=U.shape[0], rank=U.shape[1], residual=True, l1=0.0)
        model = LowRankMap(cfg); model.load_state_dict(sd)
    elif meta["model"] == "linear":
        W = sd["W.weight"]
        model = LinearMap(W.shape[1], W.shape[0]); model.load_state_dict(sd)
    else:
        w0 = sd["net.0.weight"]; w2 = sd["net.2.weight"]
        model = TinyMLP(w0.shape[1], w2.shape[0], hidden=w0.shape[0]); model.load_state_dict(sd)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--text", type=str, default="The capital of France is")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    A = load_model(cfg["models"]["A"]["name"])
    B = load_model(cfg["models"]["B"]["name"])
    kind = cfg["layers"]["hook_point"]

    model = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get A direction from penultimate token
    hn = hook_name(args.layer, kind)
    toksA = A.tokenizer(args.text, return_tensors="pt").to(A.model.cfg.device)
    _, cacheA = A.model.run_with_cache(toksA, names_filter=[hn])
    Aact = cacheA[hn][0].detach()  # [seq, d_in]
    directionA = Aact[-2]  # [d_in]

    with torch.no_grad():
        directionB = model(directionA.unsqueeze(0).to(device)).squeeze(0).detach().cpu()  # [d_out]

    # Inject into B at same token position
    toksB = B.tokenizer(args.text, return_tensors="pt").to(B.model.cfg.device)
    pos = toksB.input_ids.shape[-1]-2

    def hook_fn(act, hook):
        act[pos, :] = act[pos, :] + args.alpha * directionB.to(act.device)
        return act

    with B.model.hook_points[hn].register_hook(hook_fn):
        patched_logits = B.model(toksB).logits
    base_logits = B.model(toksB).logits

    def topk(logits, k=5):
        probs = torch.softmax(logits[0, -1], dim=-1)
        v, i = torch.topk(probs, k)
        toks = [B.tokenizer.decode([ii.item()]) for ii in i]
        return list(zip(toks, [x.item() for x in v]))

    print("BASE:", topk(base_logits))
    print("PATCHED:", topk(patched_logits))

if __name__ == "__main__":
    main()
