import numpy as np

CENTER_PATH = "/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/features/layer4_centers.npy"
FEATURE_PATH = "/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/feature_extraction/output/layer4/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.npy"

centers = np.load(CENTER_PATH, allow_pickle=True).item()
feat = np.load(FEATURE_PATH)
feat = np.squeeze(feat)

# ✅ 모든 경우를 포괄하는 flatten 방식
if feat.ndim == 4:
    feat = feat.reshape(feat.shape[1], -1).mean(axis=1)  # (C,H,W) -> (C,)
elif feat.ndim == 3:
    feat = feat.reshape(feat.shape[0], -1).mean(axis=1)
elif feat.ndim == 2:
    feat = feat.mean(axis=1)
elif feat.ndim == 1:
    pass
else:
    raise ValueError(f"Unexpected feature shape: {feat.shape}")

feat = feat / (np.linalg.norm(feat) + 1e-8)

# ✅ 클래스 이름 정리
target_cls = "applyeyemakeup"
center = centers[target_cls]
center = center / (np.linalg.norm(center) + 1e-8)

# ✅ 차원 검증
print("Center shape:", center.shape)
print("Feature shape:", feat.shape)
assert center.shape == feat.shape, "❌ Feature and center dimension mismatch!"

# ✅ 유사도 계산
sim = np.dot(center, feat)
print("Cosine similarity:", sim)
print("Mean:", np.mean(feat), "Std:", np.std(feat))