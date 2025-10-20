import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# ==============================
# Config
# ==============================
LAYER = "layer3"
FEATURE_DIR = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/feature_extraction/output/{LAYER}"
CENTER_PATH = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/features/{LAYER}_centers.npy"
SAVE_PATH = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/features/{LAYER}_tau_exit_acc_based.npy"

TARGET_PRECISION = 0.95   # 목표 edge exit 정확도

# ==============================
# Semantic Center 로드
# ==============================
centers = np.load(CENTER_PATH, allow_pickle=True).item()
classes = sorted(list(centers.keys()))
print(f"🔹 Loaded semantic centers for {len(classes)} classes from {CENTER_PATH}")

scores, labels = [], []  # ΔS값과 정답여부 저장

# ==============================
# ΔS 계산 루프
# ==============================
for cls in tqdm(classes, desc=f"Computing ΔS for {LAYER}"):

    # 폴더 이름 매칭
    class_dir = None
    for d in os.listdir(FEATURE_DIR):
        clean_d = d.lower().replace(" ", "").replace("_", "")
        clean_cls = cls.lower().replace(" ", "").replace("_", "")
        if clean_d == clean_cls:
            class_dir = os.path.join(FEATURE_DIR, d)
            break
    if not class_dir or not os.path.isdir(class_dir):
        continue

    files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
    if len(files) < 2:
        continue

    # 모든 클래스 semantic center 행렬 구성
    center_matrix = np.stack(
        [centers[c] / (np.linalg.norm(centers[c]) + 1e-8) for c in classes], axis=0
    )

    for file in files:
        feat = np.load(os.path.join(class_dir, file))
        feat = np.squeeze(feat)

        # Feature 정규화
        if feat.ndim == 4:  # (1, C, H, W)
            feat = torch.tensor(feat)
            feat = F.adaptive_avg_pool2d(feat, 1).view(-1).numpy()
        elif feat.ndim == 3:
            feat = feat.mean(axis=(1, 2))
        elif feat.ndim == 2:
            feat = feat.mean(axis=1)
        feat = feat / (np.linalg.norm(feat) + 1e-8)

        # cosine similarity
        sims = center_matrix @ feat
        sort_s = np.sort(sims)
        top1, top2 = sort_s[-1], sort_s[-2]

        # ΔS 계산
        delta_s = (top1 - top2) / (abs(top1) + abs(top2) + 1e-8)

        scores.append(delta_s)
        labels.append(int(np.argmax(sims) == classes.index(cls)))

scores = np.asarray(scores)
labels = np.asarray(labels)
assert len(scores) == len(labels) and len(scores) > 0, "❌ ΔS 계산 실패 또는 데이터 부족"

# ==============================
# 정확도 기반 τ 탐색
# ==============================
sorted_scores = np.sort(np.unique(scores))
best_tau = sorted_scores[-1]
for tau in sorted_scores:
    mask = scores >= tau
    if mask.sum() == 0:
        continue
    prec = labels[mask].mean()
    if prec >= TARGET_PRECISION:
        best_tau = tau
        break

exit_rate = (scores >= best_tau).mean()
edge_acc = labels[scores >= best_tau].mean()

print("\n✅ Accuracy-based τ calibration 완료")
print(f"Target precision ≥ {TARGET_PRECISION*100:.0f}%")
print(f"Calibrated τ_exit = {best_tau:.3f}")
print(f"Exit rate = {exit_rate*100:.2f}%, Edge exit accuracy = {edge_acc*100:.2f}%")

# ==============================
# Plot 시각화
# ==============================
bins = np.linspace(0, 1, 200)
plt.figure(figsize=(8, 5))
plt.hist(scores[labels == 1], bins=bins, alpha=0.6, label="Correct ΔS", color="skyblue")
plt.hist(scores[labels == 0], bins=bins, alpha=0.6, label="Wrong ΔS", color="salmon")

plt.axvline(best_tau, color="black", linestyle="--", linewidth=1.5,
            label=f"τ_exit={best_tau:.3f} (≥{TARGET_PRECISION*100:.0f}% acc)")
plt.xlabel("ΔS = S_top1 - S_top2", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.title(f"ΔS-based τ_exit estimation (Accuracy-calibrated, {LAYER})", fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()

plot_path = f"{os.path.dirname(SAVE_PATH)}/{LAYER}_deltaS_accCalib_plot.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"📊 Accuracy-based τ_exit plot saved at: {plot_path}")
np.save(SAVE_PATH, {"tau_exit": best_tau, "target_precision": TARGET_PRECISION})