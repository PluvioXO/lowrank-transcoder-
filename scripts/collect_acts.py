import os, argparse, yaml, torch
from src.utils import set_seed, ensure_dir, load_texts
from src.backend_tlens import load_model, get_layer_acts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    set_seed(cfg["seed"])
    A = load_model(cfg["models"]["A"]["name"])
    B = load_model(cfg["models"]["B"]["name"])
    kind = cfg["layers"]["hook_point"]

    texts_train = load_texts(cfg["data"]["dataset"], cfg["data"].get("subset"), cfg["data"]["split"], cfg["data"]["text_column"], cfg["data"]["num_docs_train"])
    texts_eval = load_texts(cfg["data"]["dataset"], cfg["data"].get("subset"), cfg["data"]["split"], cfg["data"]["text_column"], cfg["data"]["num_docs_eval"])

    outdir = os.path.join("runs", cfg["run_name"], "acts")
    ensure_dir(outdir)

    for L in args.layers:
        XA = get_layer_acts(A, texts_train, L, kind, cfg["data"]["max_length"])  # [N, dA]
        XB = get_layer_acts(B, texts_train, L, kind, cfg["data"]["max_length"])  # [N, dB]
        torch.save({"XA": XA, "XB": XB}, os.path.join(outdir, f"train_L{L}.pt"))
        XAe = get_layer_acts(A, texts_eval, L, kind, cfg["data"]["max_length"])
        XBe = get_layer_acts(B, texts_eval, L, kind, cfg["data"]["max_length"])
        torch.save({"XA": XAe, "XB": XBe}, os.path.join(outdir, f"eval_L{L}.pt"))
        print(f"[L{L}] train: A{tuple(XA.shape)} → B{tuple(XB.shape)} ; eval: A{tuple(XAe.shape)} → B{tuple(XBe.shape)}")

if __name__ == "__main__":
    main()
