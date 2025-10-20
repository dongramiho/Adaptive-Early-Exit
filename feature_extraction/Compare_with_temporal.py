# simulate_edge_exit_layer3_semantic_only.py
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

DEFAULT_TAU = 0.03
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
# Feature loop (semantic-only separability)
# ==============================
semantic_scores, is_correct_pred = [], []

for d in tqdm(sorted(os.listdir(FEATURE_DIR)), desc=f"Collecting {LAYER} semantic scores"):
    d_path = os.path.join(FEATURE_DIR, d)
    if not os.path.isdir(d_path):
        continue
    match = [c for c in classes if cls_norm(c) == cls_norm(d)]
    if not match:
        continue
    gt_cls = match[0]; gt_idx = cls2idx[gt_cls]

    files = sorted([f for f in os.listdir(d_path) if f.endswith(".npy")])
    if not files:
        continue

    for f in files:
        feat = load_vec(os.path.join(d_path, f))
        # 각 클래스 중심과의 L2 거리 계산
        sims = C @ feat
        sorted_sims = np.sort(sims)
        top1, top2 = sorted_sims[-1], sorted_sims[-2]
        S = (top1 - top2) / (abs(top1) + abs(top2) + 1e-8)
        semantic_scores.append(S)

        # ✅ 예측 일치 여부 확인 (cosine 최대값 비교)
        is_correct_pred.append(int(np.argmax(sims) == gt_idx))

semantic_scores = np.asarray(semantic_scores)
is_correct_pred = np.asarray(is_correct_pred)
total = len(semantic_scores)
assert total > 0, "No semantic scores computed."

# ==============================
# Evaluation function
# ==============================
def evaluate_at_tau(tau: float):
    exit_mask = semantic_scores >= tau
    exit_cnt = int(exit_mask.sum())
    exit_rate = exit_cnt / total
    edge_acc = float(is_correct_pred[exit_mask].mean()) if exit_cnt > 0 else float("nan")
    return {"tau": tau, "exit_rate": exit_rate, "edge_exit_accuracy": edge_acc}

# ==============================
# Fixed τ evaluation
# ==============================
base = evaluate_at_tau(DEFAULT_TAU)
print("\n=== Fixed-τ evaluation (Semantic-only) ===")
for k, v in base.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

# ==============================
# τ sweep
# ==============================
taus = np.linspace(semantic_scores.min(), semantic_scores.max(), 60)
exit_rates, edge_accs = [], []
for t in taus:
    m = evaluate_at_tau(float(t))
    exit_rates.append(m["exit_rate"])
    edge_accs.append(m["edge_exit_accuracy"])

# ==============================
# Plot
# ==============================
plt.figure(figsize=(10, 4.8))

# --- Left: τ vs Exit Rate & Accuracy ---
plt.subplot(1, 2, 1)
plt.plot(taus, exit_rates, label="Exit Rate", color="tab:blue", linewidth=2)
plt.plot(taus, edge_accs, label="Edge Exit Accuracy", color="tab:green", linewidth=2)
plt.axvline(DEFAULT_TAU, color="black", linestyle="--", label=f"τ={DEFAULT_TAU:.3f}", linewidth=1.5)
plt.scatter(DEFAULT_TAU, base["exit_rate"], color="tab:blue", s=60, zorder=5)
plt.scatter(DEFAULT_TAU, base["edge_exit_accuracy"], color="tab:green", s=60, zorder=5)
plt.xlim(0, 0.1)
plt.xlabel("τ_exit (Threshold)", fontsize=11)
plt.ylabel("Rate / Accuracy", fontsize=11)
plt.title("Exit Rate & Edge Accuracy vs τ (Semantic-only)", fontsize=12, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

# --- Right: Trade-off Curve ---
plt.subplot(1, 2, 2)
plt.plot(exit_rates, edge_accs, marker="o", color="#3A86FF", linewidth=2)
plt.scatter(base["exit_rate"], base["edge_exit_accuracy"], color="red", s=70, label="Fixed τ point")
plt.text(base["exit_rate"] + 0.005, base["edge_exit_accuracy"],
         f"{base['edge_exit_accuracy']*100:.2f}%", fontsize=9, fontweight="bold")
plt.xlabel("Exit Rate (fraction of early exits)", fontsize=11)
plt.ylabel("Edge Exit Accuracy", fontsize=11)
plt.title("Semantic-only Trade-off Curve", fontsize=12, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, f"{LAYER}_semantic_only_exit_tradeoff.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"📊 plot saved at: {plot_path}")

# ==============================
# Summary
# ==============================
print("\n=== Semantic-only τ evaluation summary ===")
print(f"τ (default): {DEFAULT_TAU:.3f}")
print(f"Exit Rate: {base['exit_rate']*100:.2f}%")
print(f"Edge Exit Accuracy: {base['edge_exit_accuracy']*100:.2f}%")