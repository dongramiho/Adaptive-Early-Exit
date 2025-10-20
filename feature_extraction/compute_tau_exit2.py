import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F

# ==============================
# Layer 설정
# ==============================
LAYER = "layer3"

FEATURE_DIR = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/feature_extraction/output/{LAYER}"
CENTER_PATH = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/features/{LAYER}_centers.npy"
SAVE_PATH = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/features/{LAYER}_tau_exit.npy"

# ==============================
# Semantic Center 로드
# ==============================
centers = np.load(CENTER_PATH, allow_pickle=True).item()
classes = sorted(list(centers.keys()))
print(f"🔹 Loaded semantic centers for {len(classes)} classes from {CENTER_PATH}")

delta_s_correct, delta_s_wrong = [], []

# ==============================
# ΔS 계산 루프
# ==============================
for cls in tqdm(classes, desc=f"Computing ΔS for {LAYER}"):

    # 폴더 이름 자동 매칭 (대소문자, 공백, 언더바 무시)
    class_dir = None
    for d in os.listdir(FEATURE_DIR):
        clean_d = d.lower().replace(" ", "").replace("_", "")
        clean_cls = cls.lower().replace(" ", "").replace("_", "")
        if clean_d == clean_cls:
            class_dir = os.path.join(FEATURE_DIR, d)
            break

    if not class_dir or not os.path.isdir(class_dir):
        print(f"⚠️ Skipping: no folder matched for class {cls}")
        continue

    files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
    if len(files) < 2:
        print(f"⚠️ Skipping {cls}: not enough samples ({len(files)} files).")
        continue

    # 모든 클래스 semantic center를 하나의 행렬로 구성
    center_matrix = np.stack(
        [centers[c] / (np.linalg.norm(centers[c]) + 1e-8) for c in classes], axis=0
    )

    for file in files:
        feat = np.load(os.path.join(class_dir, file))
        feat = np.squeeze(feat)

        # 다양한 feature shape 대응
        if feat.ndim == 4:  # (1, C, H, W)
            feat = torch.tensor(feat)
            feat = F.adaptive_avg_pool2d(feat, 1).view(-1).numpy()
        elif feat.ndim == 3:  # (C, H, W)
            feat = feat.mean(axis=(1, 2))
        elif feat.ndim == 2:
            feat = feat.mean(axis=1)
        feat = feat / (np.linalg.norm(feat) + 1e-8)

        # cosine similarity 계산
        sims = center_matrix @ feat
        sorted_sims = np.sort(sims)
        top1, top2 = sorted_sims[-1], sorted_sims[-2]

        # 🔹 정규화된 ΔS 계산 (COACH + 보정)
        delta_s = (top1 - top2) / (abs(top1) + abs(top2) + 1e-8)
        delta_s_correct.append(delta_s if np.argmax(sims) == classes.index(cls) else -delta_s)

# ==============================
# ΔS 분포 스케일 보정 (Z-score + 0~1 정규화)
# ==============================
delta_s_correct = np.array(delta_s_correct)
mean_val, std_val = np.mean(delta_s_correct), np.std(delta_s_correct)
delta_s_scaled = (delta_s_correct - mean_val) / (std_val + 1e-8)
delta_s_scaled = np.clip((delta_s_scaled - np.min(delta_s_scaled)) / (np.max(delta_s_scaled) - np.min(delta_s_scaled)), 0, 1)

# Positive/Negative 분리
delta_s_pos = delta_s_scaled[delta_s_correct > 0]
delta_s_neg = delta_s_scaled[delta_s_correct < 0]

if len(delta_s_pos) == 0 or len(delta_s_neg) == 0:
    print("\n❌ ΔS 데이터가 충분하지 않습니다.")
    exit()

# ==============================
# ΔS 분포 교차점 계산
# ==============================
bins = np.linspace(0, 1, 200)
hist_pos, _ = np.histogram(delta_s_pos, bins=bins, density=True)
hist_neg, _ = np.histogram(np.abs(delta_s_neg), bins=bins, density=True)
diff = np.abs(hist_pos - hist_neg)
idx = np.argmin(diff)
tau_exit = bins[idx]

np.save(SAVE_PATH, {"tau_exit": tau_exit})
print(f"\n✅ Estimated τ_exit (ΔS-based, scaled) = {tau_exit:.3f}")
print(f"ΔS_pos (mean={np.mean(delta_s_pos):.4f}, std={np.std(delta_s_pos):.4f})")
print(f"ΔS_neg (mean={np.mean(delta_s_neg):.4f}, std={np.std(delta_s_neg):.4f})")

# ==============================
# Plot
# ==============================
plt.figure(figsize=(8, 5))
plt.hist(delta_s_pos, bins=bins, alpha=0.6, label="Correct Exit ΔS", color="skyblue")
plt.hist(np.abs(delta_s_neg), bins=bins, alpha=0.6, label="Wrong Exit ΔS", color="salmon")
plt.axvline(tau_exit, color="black", linestyle="--", label=f'τ_exit={tau_exit:.3f}')
plt.xlabel("Scaled ΔS (normalized separability)")
plt.ylabel("Density")
plt.legend()
plt.title(f"ΔS-based τ_exit estimation (scaled) — {LAYER}")
plt.tight_layout()

plot_path = f"{os.path.dirname(SAVE_PATH)}/{LAYER}_deltaS_tau_exit_scaled_plot.png"
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"📊 τ_exit plot saved at: {plot_path}")