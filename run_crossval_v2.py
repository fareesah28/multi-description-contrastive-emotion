import argparse, subprocess, time

# paths
VIDEO_DIR = "/mnt/scratch/users/adhg808/MAFW/data/clips"
CSV_DIR   = "/mnt/scratch/users/adhg808/MAFW/csv/folds"

ap = argparse.ArgumentParser()
ap.add_argument("--caption_pool", choices=["mean_token","eot_token"], default="mean_token")
ap.add_argument("--temporal_layers", type=int, default=1)
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument(
    "--tune_policy",
    choices=["light","strict","vision_unfrozen","text_unfrozen","all_unfrozen"],
    default="light"
)
ap.add_argument("--logit_temp", action="store_true", help="Use learnable temperature (logit_scale).")
args = ap.parse_args()

fold_list = [int(x) for x in args.folds.split(",")]

for fold in fold_list:
    print(f"\n Starting Fold {fold} | pool={args.caption_pool} | T={args.temporal_layers} | policy={args.tune_policy} | lt={args.logit_temp}")
    start_time = time.time()
    cmd = [
        "python", "train_ablate.py",
        "--train_csv", f"{CSV_DIR}/fold_{fold}_train.csv",
        "--val_csv",   f"{CSV_DIR}/fold_{fold}_val.csv",
        "--video_dir", VIDEO_DIR,
        "--fold", str(fold),
        "--caption_pool", args.caption_pool,
        "--temporal_layers", str(args.temporal_layers),
        "--tune_policy", args.tune_policy
    ]
    if args.logit_temp:
        cmd.append("--logit_temp")

    subprocess.run(cmd, check=True)
    print(f" Finished Fold {fold} in {time.time() - start_time:.1f} seconds")

