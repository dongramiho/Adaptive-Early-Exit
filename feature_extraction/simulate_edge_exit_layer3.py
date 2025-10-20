# simulate_edge_exit_layer3.py
import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==============================
# Config
# ==============================
LAYER = "layer3"
FEATURE_DIR = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/feature_extraction/output/{LAYER}"
CENTER_PATH = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/features/{LAYER}_centers.npy"

ALPHA = 0.9
BETA  = 0.1
DEFAULT_TAU = 0.1018

OUT_DIR = "/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/features/edge_exit_eval"
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================
# Helpers
# ==============================
def cls_norm(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "")

def load_vec(path: str) -> np.ndarray:
    x = np.load(path); x = np.squeeze(x)
    if x.ndim == 4:
        x = torch.tensor(x)
        x = F.adaptive_avg_pool2d(x, 1).view(-1).numpy()
    elif x.ndim == 3:
        x = x.mean(axis=(1, 2))
    elif x.ndim == 2:
        x = x.mean(axis=1)
    x = x / (np.linalg.norm(x) + 1e-8)
    return x

# ==============================
# Load centers
# ==============================
centers = np.load(CENTER_PATH, allow_pickle=True).item()
classes = sorted(list(centers.keys()))
cls2idx = {c:i for i,c in enumerate(classes)}
C = np.stack([centers[c] / (np.linalg.norm(centers[c]) + 1e-8) for c in classes])
print(f"🔹 Loaded {len(classes)} centers from {CENTER_PATH}")

# ==============================
# Feature loop
# ==============================
hybrid_scores, is_correct_pred, per_sample_cls = [], [], []
for d in tqdm(sorted(os.listdir(FEATURE_DIR)), desc=f"Collecting {LAYER} scores"):
    d_path = os.path.join(FEATURE_DIR, d)
    if not os.path.isdir(d_path): continue
    match = [c for c in classes if cls_norm(c) == cls_norm(d)]
    if not match: continue
    gt_cls = match[0]; gt_idx = cls2idx[gt_cls]

    files = sorted([f for f in os.listdir(d_path) if f.endswith(".npy")])
    if not files: continue

    prev_feat = None
    for f in files:
        feat = load_vec(os.path.join(d_path, f))
        sims = C @ feat
        sort_s = np.sort(sims)
        top1, top2 = sort_s[-1], sort_s[-2]
        delta_s = (top1 - top2) / (abs(top1) + abs(top2) + 1e-8)
        if prev_feat is None:
            s_temp = 0.0
        else:
            s_temp = np.clip(float(np.dot(prev_feat, feat)), 0.0, 1.0)
        prev_feat = feat

        hybrid = ALPHA * delta_s + BETA * s_temp
        hybrid_scores.append(hybrid)
        is_correct_pred.append(int(np.argmax(sims) == gt_idx))
        per_sample_cls.append(gt_idx)

hybrid_scores = np.asarray(hybrid_scores)
is_correct_pred = np.asarray(is_correct_pred)
total = len(hybrid_scores)
assert total > 0, "No samples collected."

# ==============================
# Evaluation function
# ==============================
def evaluate_at_tau(tau: float):
    exit_mask = hybrid_scores >= tau
    exit_cnt = int(exit_mask.sum())
    exit_rate = exit_cnt / total
    edge_acc = float(is_correct_pred[exit_mask].mean()) if exit_cnt > 0 else float("nan")
    overall_correct = is_correct_pred[exit_mask].sum() + (~exit_mask).sum()
    overall_acc = overall_correct / total
    return {
        "tau": tau,
        "exit_rate": exit_rate,
        "edge_exit_accuracy": edge_acc,
        "overall_accuracy_oracle": overall_acc,
        "exit_count": exit_cnt,
        "total": total,
    }

# ==============================
# 3) Fixed τ evaluation
# ==============================
base = evaluate_at_tau(DEFAULT_TAU)
print("\n=== Fixed-τ evaluation (Layer3) ===")
for k,v in base.items():
    if isinstance(v,float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

# ==============================
# 4) τ sweep
# ==============================
taus = np.linspace(hybrid_scores.min(), hybrid_scores.max(), 60)
exit_rates, edge_accs, overall_accs = [], [], []

for t in taus:
    m = evaluate_at_tau(float(t))
    exit_rates.append(m["exit_rate"])
    edge_accs.append(m["edge_exit_accuracy"])
    overall_accs.append(m["overall_accuracy_oracle"])

best_idx = int(np.argmax(overall_accs))
best_tau = float(taus[best_idx])

print("\n=== τ sweep summary ===")
print(f"best_tau (by overall acc): {best_tau:.4f}")
print(f"overall_acc: {overall_accs[best_idx]:.4f}, exit_rate: {exit_rates[best_idx]:.4f}, edge_acc: {edge_accs[best_idx]:.4f}")

# ==============================
# 5) Plot: Exit Rate & Accuracy Trade-off (Server 제외)
# ==============================
plt.figure(figsize=(10, 4.8))

# --- Left: τ vs Exit Rate & Edge Accuracy ---
plt.subplot(1, 2, 1)
plt.plot(taus, exit_rates, label="Exit Rate", color="tab:blue", linewidth=2)
plt.plot(taus, edge_accs, label="Edge Exit Accuracy", color="tab:green", linewidth=2)
plt.axvline(DEFAULT_TAU, color="black", linestyle="--", label=f"τ={DEFAULT_TAU:.3f}", linewidth=1.5)

# 포인트 표시
plt.scatter(DEFAULT_TAU, base["exit_rate"], color="tab:blue", s=60, zorder=5)
plt.scatter(DEFAULT_TAU, base["edge_exit_accuracy"], color="tab:green", s=60, zorder=5)

# 텍스트 주석
plt.text(DEFAULT_TAU + 0.002, base["exit_rate"], f"{base['exit_rate']*100:.1f}%", fontsize=8)
plt.text(DEFAULT_TAU + 0.002, base["edge_exit_accuracy"] - 0.02, f"{base['edge_exit_accuracy']*100:.2f}%", fontsize=8)

plt.xlabel("τ_exit (Threshold)", fontsize=11)
plt.ylabel("Rate / Accuracy", fontsize=11)
plt.title("Exit Rate & Edge Accuracy vs τ", fontsize=12, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

# --- Right: Exit Rate vs Edge Accuracy Trade-off ---
plt.subplot(1, 2, 2)
plt.plot(exit_rates, edge_accs, marker="o", color="#3A86FF", linewidth=2)
plt.scatter(base["exit_rate"], base["edge_exit_accuracy"], color="red", s=70, label="Fixed τ point")

# 값 강조
plt.text(base["exit_rate"] + 0.005, base["edge_exit_accuracy"],
         f"{base['edge_exit_accuracy']*100:.2f}%", fontsize=9, fontweight="bold")

plt.xlabel("Exit Rate (fraction of early exits)", fontsize=11)
plt.ylabel("Edge Exit Accuracy", fontsize=11)
plt.title("Edge Trade-off Curve", fontsize=12, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()

plot_path = os.path.join(OUT_DIR, f"{LAYER}_edge_exit_accuracy_tradeoff.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"📊 plot saved at: {plot_path}")

# ==============================
# 결과 요약 출력
# ==============================
print(f"\n=== Edge-only τ evaluation summary ===")
print(f"τ (default): {DEFAULT_TAU:.3f}")
print(f"Exit Rate: {base['exit_rate']*100:.2f}%")
print(f"Edge Exit Accuracy: {base['edge_exit_accuracy']*100:.2f}%")
print(f"Best τ (sweep): {best_tau:.3f} → Exit Rate {exit_rates[best_idx]*100:.2f}%, "
      f"Edge Acc {edge_accs[best_idx]*100:.2f}%")