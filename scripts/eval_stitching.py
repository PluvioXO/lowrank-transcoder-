\
import os, argparse, yaml, torch
import torch.nn.functional as F
from src.utils import set_seed, load_texts
from src.backend_tlens import load_model, stitch_with_mapping
from src.cka_svcca import linear_cka, svcca
from src.transcoder import TranscoderConfig, LowRankMap, LinearMap, TinyMLP

def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu")
    meta = ck["meta"]; sd = ck["state_dict"]
    model_type = meta["model"]
    if model_type == "lowrank":
        U = sd["U"]; V = sd["V"]
        cfg = TranscoderConfig(d_in=V.shape[0], d_out=U.shape[0], rank=U.shape[1], residual=True, l1=0.0)
        model = LowRankMap(cfg); model.load_state_dict(sd)
    elif model_type == "linear":
        # weight name: 'W.weight' [d_out, d_in]
        W = sd["W.weight"]
        model = LinearMap(W.shape[1], W.shape[0]); model.load_state_dict(sd)
    else:
        # infer hidden from keys
        w0 = sd["net.0.weight"]; w2 = sd["net.2.weight"]
        model = TinyMLP(w0.shape[1], w2.shape[0], hidden=w0.shape[0]); model.load_state_dict(sd)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])

    # load eval activations
    d = torch.load(os.path.join("runs", cfg["run_name"], "acts", f"eval_L{args.layer}.pt"), map_location="cpu")
    XAe, XBe = d["XA"], d["XB"]

    model = load_checkpoint(args.checkpoint)
    with torch.no_grad():
        pred = model(XAe)
        mse = F.mse_loss(pred, XBe).item()
        cka = linear_cka(pred, XBe)
        cca = svcca(pred, XBe, n_comp=min(64, XAe.shape[1], XBe.shape[1]))
    print(f"Eval: MSE={mse:.6f}, CKA={cka:.4f}, SVCCA={cca:.4f}")

    # stitched loss
    A = load_model(cfg["models"]["A"]["name"])
    B = load_model(cfg["models"]["B"]["name"])
    eval_texts = load_texts(cfg["data"]["dataset"], cfg["data"].get("subset"), cfg["data"]["split"], cfg["data"]["text_column"], cfg["data"]["num_docs_eval"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def mapper_fn(Aact):
        return model(Aact.to(device)).detach().cpu()

    ppl = stitch_with_mapping(A, B, args.layer, cfg["layers"]["hook_point"], mapper_fn, eval_texts, cfg["data"]["max_length"], max_batches=cfg["stitching_eval"]["max_batches"])
    print(f"Stitched NLL (approx): {ppl:.4f}")

if __name__ == "__main__":
    main()
