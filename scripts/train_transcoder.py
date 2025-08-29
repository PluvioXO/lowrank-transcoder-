\
import os, argparse, yaml, torch
import torch.nn.functional as F
from tqdm import tqdm
from src.utils import set_seed, ensure_dir
from src.transcoder import TranscoderConfig, LowRankMap, LinearMap, TinyMLP

def load_pair(run_dir, L):
    return torch.load(os.path.join(run_dir, "acts", f"train_L{L}.pt"), map_location="cpu")

def eval_pair(run_dir, L):
    return torch.load(os.path.join(run_dir, "acts", f"eval_L{L}.pt"), map_location="cpu")

def build_model(kind: str, d_in: int, d_out: int, rank: int, l1: float, residual: bool, hidden: int):
    if kind == "lowrank":
        cfg = TranscoderConfig(d_in=d_in, d_out=d_out, rank=rank, residual=residual, l1=l1, prox_every=50)
        return LowRankMap(cfg)
    elif kind == "linear":
        return LinearMap(d_in, d_out)
    else:
        return TinyMLP(d_in, d_out, hidden=hidden)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--model", type=str, choices=["lowrank","linear","mlp"], required=True)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--l1", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=512)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])

    run_dir = os.path.join("runs", cfg["run_name"])
    ensure_dir(run_dir)
    dtrain = load_pair(run_dir, args.layer)
    dval = eval_pair(run_dir, args.layer)

    XA, XB = dtrain["XA"], dtrain["XB"]
    XAe, XBe = dval["XA"], dval["XB"]
    d_in, d_out = XA.shape[1], XB.shape[1]

    model = build_model(args.model, d_in, d_out, args.rank, args.l1,
                        residual=cfg["lowrank"]["residual"], hidden=args.hidden)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    XA, XB = XA.to(device), XB.to(device)
    XAe, XBe = XAe.to(device), XBe.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    steps = cfg["train"]["steps"]
    bs = cfg["train"]["batch_size_pairs"]
    eval_every = cfg["train"]["eval_every"]

    def iterate_pairs(XA, XB, bs):
        N = XA.shape[0]
        idx = torch.randperm(N, device=XA.device)
        for i in range(0, N, bs):
            j = idx[i:i+bs]
            yield XA[j], XB[j]

    it = iterate_pairs(XA, XB, bs)
    for step in range(1, steps+1):
        try:
            xa, xb = next(it)
        except StopIteration:
            it = iterate_pairs(XA, XB, bs)
            xa, xb = next(it)
        pred = model(xa); loss = F.mse_loss(pred, xb)
        if hasattr(model, "l1_penalty"):
            loss = loss + model.l1_penalty()
        opt.zero_grad(); loss.backward(); opt.step()
        if hasattr(model, "prox") and model.cfg.prox_every and (step % model.cfg.prox_every == 0):
            model.prox(t=cfg["train"]["lr"])

        if step % eval_every == 0:
            with torch.no_grad():
                mse = F.mse_loss(model(XAe), XBe).item()
            print(f"[{step}/{steps}] train_mse={loss.item():.6f} eval_mse={mse:.6f}")

    ckpt = os.path.join(run_dir, f"{args.model}_L{args.layer}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "meta": {"model": args.model, "layer": args.layer, "d_in": d_in, "d_out": d_out,
                 "rank": getattr(model, 'cfg', None).rank if hasattr(model, 'cfg') else None,
                 "hidden": args.hidden if args.model=='mlp' else None}
    }, ckpt)
    print("Saved", ckpt)

if __name__ == "__main__":
    main()
