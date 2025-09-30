import argparse, glob, json, numpy as np, os, re
from collections import defaultdict

METRICS = ["val_recall@1", "val_recall@5", "val_recall@10", "val_cosine_sim"]

def load_rows(pattern):
    files = sorted(glob.glob(pattern))
    rows = []
    for f in files:
        try:
            rows.append((f, json.load(open(f))))
        except Exception as e:
            print(f"Skip {f}: {e}")
    return rows

def mm_std(values):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())

def summarize_set(rows, label):
    if not rows:
        print(f"\n {label} \nNo files found.")
        return
    print(f"\n {label} ({len(rows)} folds) ")
    # per-fold quick view
    fold_re = re.compile(r"fold_(\d+)")
    per_fold = defaultdict(dict)
    for f, r in rows:
        m = fold_re.search(os.path.basename(f))
        fold = int(m.group(1)) if m else None
        if fold is not None:
            per_fold[fold] = r
    if per_fold:
        print("Per-fold:")
        header = " fold " + "".join([f"{k:>15}" for k in METRICS])
        print(header)
        for fold in sorted(per_fold.keys()):
            r = per_fold[fold]
            line = f"{fold:5d}" + "".join([f"{r[k]:15.4f}" for k in METRICS])
            print(line)

    # mean ± std
    print("\nSummary:")
    for k in METRICS:
        m, s = mm_std([r[k] for _, r in rows])
        print(f"{k:>14}: {m:.4f} ± {s:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption_pool", default="mean_token",
                    choices=["mean_token", "eot_token"],
                    help="Which caption pooling the runs used.")
    ap.add_argument("--temporal_layers", nargs="+", default=["1","2"],
                    help="List of temporal depths to summarize, e.g. 1 2 3.")
    ap.add_argument("--metrics_dir", default="metrics",
                    help="Directory where fold_*.json files are saved.")
    ap.add_argument("--prefix", default="",
                    help="Optional metrics filename prefix, e.g. 'strict_' or 'light_'.")
    ap.add_argument("--suffix", default="",
                    help="Optional metrics filename suffix, e.g. '_lt'.")
    args = ap.parse_args()

    for T in args.temporal_layers:
        pattern = os.path.join(
            args.metrics_dir,
            f"{args.prefix}fold_*_pool-{args.caption_pool}_T{T}{args.suffix}.json"
        )
        rows = load_rows(pattern)
        label_prefix = args.prefix if args.prefix else "no-prefix"
        label_suffix = args.suffix if args.suffix else "no-suffix"
        summarize_set(rows, label=f"{label_prefix} | {args.caption_pool} | T={T} | {label_suffix}")

if __name__ == "__main__":
    main()
