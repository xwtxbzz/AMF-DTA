import json, os, glob

print("=== Davis DrugMGR Completed Results ===")
for s in [42, 52, 62, 72, 82]:
    f = f"outputs/davis_drugmgr_seed{s}_200/test_metrics.json"
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"  seed{s}: RMSE={d['rmse']:.4f}, CI={d['ci']:.4f}, R2={d['r2']:.4f}")

print("\n=== Current Running Progress ===")
for f in sorted(glob.glob("logs/2024_*.log")):
    if "batch" in f:
        continue
    lines = open(f).read().strip().split("\n")
    epoch_count = sum(1 for l in lines if "'epoch'" in l)
    print(f"  {os.path.basename(f)}: {epoch_count} epochs")

print("\n=== Output Dir Counts ===")
# 仅剩仍在仓库中的模型；其余模型已移除
for model in ["drugmgr"]:
    count = len(glob.glob(f"outputs/*{model}_seed*_200"))
    print(f"  {model}: {count}/15 complete")
