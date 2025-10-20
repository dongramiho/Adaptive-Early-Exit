import os, numpy as np, torch, torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


LAYER = "layer3"
FEATURE_DIR = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/feature_extraction/output/{LAYER}"
CENTER_PATH = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/features/{LAYER}_centers.npy"
TARGET_PREC = 0.95     # 원하는 edge exit accuracy (precision)
ALPHA, BETA = 0.9, 0.1

def cls_norm(s): return s.lower().replace(" ","").replace("_","")
def load_vec(p):
    x = np.load(p); x = np.squeeze(x)
    if x.ndim==4: x=torch.tensor(x); x=F.adaptive_avg_pool2d(x,1).view(-1).numpy()
    elif x.ndim==3: x=x.mean(axis=(1,2))
    elif x.ndim==2: x=x.mean(axis=1)
    x = x/(np.linalg.norm(x)+1e-8); return x

centers = np.load(CENTER_PATH, allow_pickle=True).item()
classes = sorted(centers.keys()); idx = {c:i for i,c in enumerate(classes)}
C = np.stack([centers[c]/(np.linalg.norm(centers[c])+1e-8) for c in classes])

scores, labels = [], []  # hybrid score, 1(correct on edge)/0(wrong on edge)

for d in sorted(os.listdir(FEATURE_DIR)):
    dpath = os.path.join(FEATURE_DIR, d)
    if not os.path.isdir(dpath): continue
    m = [c for c in classes if cls_norm(c)==cls_norm(d)]
    if not m: continue
    gt = idx[m[0]]

    prev = None
    for f in sorted([f for f in os.listdir(dpath) if f.endswith(".npy")]):
        v = load_vec(os.path.join(dpath,f))
        sims = C @ v
        s = np.sort(sims); top1, top2 = s[-1], s[-2]
        delta_s = (top1 - top2) / (abs(top1)+abs(top2)+1e-8)
        s_temp  = 0.0 if prev is None else float(np.clip(np.dot(prev, v), 0.0, 1.0))
        prev = v
        hybrid = ALPHA*delta_s + BETA*s_temp
        scores.append(hybrid)
        labels.append(int(np.argmax(sims)==gt))

scores = np.asarray(scores); labels = np.asarray(labels)
# τ를 오름차순으로 훑으며 precision≥TARGET을 처음 만족하는 최소 τ 선택
order = np.argsort(scores)
unique_scores = np.unique(scores[order])

best_tau = unique_scores[-1]  # fallback (아무것도 못 찾으면 아주 보수적)
for tau in unique_scores:
    exit_mask = scores >= tau
    if exit_mask.sum()==0: continue
    prec = labels[exit_mask].mean()
    if prec >= TARGET_PREC:
        best_tau = tau
        break

print(f"✅ Calibrated τ (precision≥{TARGET_PREC*100:.0f}%) = {best_tau:.3f}")

# === 기존 코드 이후에 추가 ===
bins = np.linspace(scores.min(), scores.max(), 200)
hist_pos, _ = np.histogram(scores[np.array(labels)==1], bins=bins, density=True)
hist_neg, _ = np.histogram(scores[np.array(labels)==0], bins=bins, density=True)

plt.figure(figsize=(8, 5))
plt.hist(
    scores[np.array(labels)==1], bins=bins, alpha=0.6,
    label="Correct Hybrid Score", color="skyblue"
)
plt.hist(
    scores[np.array(labels)==0], bins=bins, alpha=0.6,
    label="Wrong Hybrid Score", color="salmon"
)
plt.axvline(best_tau, color="black", linestyle="--", linewidth=2,
            label=f"τ_exit={best_tau:.3f}")

# 🔹 텍스트 표시 (정확도 기준으로 calibrate 됨)
plt.text(
    best_tau + 0.01,
    max(hist_pos.max(), hist_neg.max()) * 0.8,
    f"Precision ≥ {TARGET_PREC*100:.0f}%\nτ={best_tau:.3f}",
    fontsize=11, fontweight="bold",
    color="black",
    bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.3")
)

plt.xlabel("Hybrid Score (α·ΔS + β·Temporal)")
plt.ylabel("Density")
plt.title(f"Hybrid Early-Exit Calibration ({LAYER}) [Precision-based τ]")
plt.legend()
plt.tight_layout()

plot_path = f"/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/features/{LAYER}_hybrid_exit_precision_based.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"📊 Precision-calibrated τ plot saved at: {plot_path}")