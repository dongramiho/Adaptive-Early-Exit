import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==============================
# Config
# ==============================
LAYER = "layer3"
FEATURE_DIR = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/feature_extraction/output/{LAYER}"
CENTER_PATH = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/features/{LAYER}_centers.npy"
SAVE_PATH = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/features/{LAYER}_hybrid_exit_v2.npy"

ALPHA = 0.9   # ΔS 비중 강화
BETA = 0.1   # Temporal similarity 비중 약화

# ==============================
# Semantic Center 로드
# ==============================
centers = np.load(CENTER_PATH, allow_pickle=True).item()
classes = sorted(list(centers.keys()))
print(f"🔹 Loaded {len(classes)} semantic centers from {CENTER_PATH}")

delta_s_vals, temporal_vals, hybrid_correct, hybrid_wrong = [], [], [], []

# ==============================
# ΔS & Temporal Consistency 계산
# ==============================
for cls in tqdm(classes, desc=f"Processing {LAYER} hybrid exit"):
    class_dir = None
    for d in os.listdir(FEATURE_DIR):
        clean_d = d.lower().replace(" ", "").replace("_", "")
        if clean_d == cls.lower().replace(" ", "").replace("_", ""):
            class_dir = os.path.join(FEATURE_DIR, d)
            break
    if not class_dir or not os.path.isdir(class_dir):
        continue

    files = sorted([f for f in os.listdir(class_dir) if f.endswith(".npy")])
    if len(files) < 3:
        continue

    center_matrix = np.stack(
        [centers[c] / (np.linalg.norm(centers[c]) + 1e-8) for c in classes],
        axis=0
    )

    prev_feat = None
    for file in files:
        feat = np.load(os.path.join(class_dir, file))
        feat = np.squeeze(feat)
        if feat.ndim == 4:
            feat = torch.tensor(feat)
            feat = F.adaptive_avg_pool2d(feat, 1).view(-1).numpy()
        elif feat.ndim == 3:
            feat = feat.mean(axis=(1, 2))
        elif feat.ndim == 2:
            feat = feat.mean(axis=1)
        feat = feat / (np.linalg.norm(feat) + 1e-8)

        # ==========================
        # ΔS 계산
        # ==========================
        sims = center_matrix @ feat
        sorted_sims = np.sort(sims)
        top1, top2 = sorted_sims[-1], sorted_sims[-2]
        delta_s = (top1 - top2) / (abs(top1) + abs(top2) + 1e-8)
        delta_s_vals.append(delta_s)

        # ==========================
        # Temporal Similarity
        # ==========================
        if prev_feat is not None:
            s_temp = float(np.dot(prev_feat, feat))
            # 0.9~1 사이의 작은 변화 구분을 위해 정규화
            s_temp = max(0.0, min(1.0, s_temp))
            s_temp = (s_temp - 0.9) / 0.1   # 0.9 → 0, 1.0 → 1.0
            s_temp = np.clip(s_temp, 0, 1)
            temporal_vals.append(s_temp)
        else:
            s_temp = 0.0
        prev_feat = feat

        # ==========================
        # ΔS / Temporal 정규화
        # ==========================
        delta_s_norm = np.clip((delta_s - 0) / (0.05 + 1e-8), 0, 1)  # ΔS는 작은 범위이므로 확장
        hybrid_score = ALPHA * delta_s_norm + BETA * s_temp

        # ==========================
        # Correct / Wrong 분류
        # ==========================
        correct_pred = np.argmax(sims) == classes.index(cls)
        if correct_pred:
            hybrid_correct.append(hybrid_score)
        else:
            hybrid_wrong.append(hybrid_score)

# ==============================
# Adaptive τ_exit 추정 (교차점)
# ==============================
if len(hybrid_correct) == 0 or len(hybrid_wrong) == 0:
    print("❌ hybrid data 부족 — class 매칭 확인 필요")
    exit()

# 분포 범위에 따라 bins 자동 설정
all_scores = np.concatenate([hybrid_correct, hybrid_wrong])
bins = np.linspace(all_scores.min(), all_scores.max(), 200)

hist_pos, _ = np.histogram(hybrid_correct, bins=bins, density=True)
hist_neg, _ = np.histogram(hybrid_wrong, bins=bins, density=True)
diff = np.abs(hist_pos - hist_neg)
idx = np.argmin(diff)
tau_exit = bins[idx]

# ==============================
# 결과 저장 및 시각화
# ==============================
np.save(SAVE_PATH, {"tau_exit": tau_exit, "alpha": ALPHA, "beta": BETA})
print(f"\n✅ Hybrid τ_exit = {tau_exit:.3f}")
print(f"ΔS mean={np.mean(delta_s_vals):.4f}, Temporal mean={np.mean(temporal_vals):.4f}")
print(f"Hybrid correct mean={np.mean(hybrid_correct):.4f}, wrong mean={np.mean(hybrid_wrong):.4f}")

plt.figure(figsize=(8, 5))
plt.hist(hybrid_correct, bins=bins, alpha=0.6, label="Correct Hybrid Score", color="skyblue")
plt.hist(hybrid_wrong, bins=bins, alpha=0.6, label="Wrong Hybrid Score", color="salmon")
plt.axvline(tau_exit, color="black", linestyle="--", label=f'τ_exit={tau_exit:.3f}')
plt.xlabel("Hybrid Score (α·ΔS + β·Temporal)")
plt.ylabel("Density")
plt.legend()
plt.title(f"Hybrid Early-Exit Estimation ({LAYER}) [Normalized]")
plt.tight_layout()

plot_path = f"{os.path.dirname(SAVE_PATH)}/{LAYER}_hybrid_exit_v2_plot.png"
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"📊 plot saved at {plot_path}")