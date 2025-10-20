import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ============================================================
# Semantic Center Construction (COACH-style)
# - Based on intra-class feature correlation filtering
# - Compatible with small per-class sample sizes (UCF101 subset)
# ============================================================

# -----------------------------
# 경로 설정
# -----------------------------
BASE_DIR = "/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/feature_extraction/output"
SAVE_DIR = "/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/features"
os.makedirs(SAVE_DIR, exist_ok=True)


def build_semantic_center_for_layer(layer_path, save_path, tau=0.7):
    """
    각 레이어에 대해 Semantic Center 계산
    - tau: intra-class correlation의 상위 비율 선택 (0~1)
    """
    semantic_centers = {}

    for class_name in tqdm(sorted(os.listdir(layer_path)), desc=f"Processing {os.path.basename(layer_path)}"):
        class_dir = os.path.join(layer_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        # 클래스명 통일 (소문자 + 언더바)
        cls_name = class_name.lower().replace(" ", "_")

        feature_list = []
        for file in os.listdir(class_dir):
            if not file.endswith(".npy"):
                continue

            feat = np.load(os.path.join(class_dir, file))
            feat = np.squeeze(feat)

            # shape 통일 (C,H,W) → 평균 Pool
            if feat.ndim == 3:
                feat = feat.mean(axis=(1, 2))
            elif feat.ndim == 2:
                feat = feat.mean(axis=1)
            elif feat.ndim == 1:
                pass
            else:
                print(f"⚠️ Unexpected shape {feat.shape} in {file}, skipping.")
                continue

            # feature 정규화 (cosine similarity 안정화)
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            feature_list.append(feat)

        # 샘플 부족한 클래스는 skip
        if len(feature_list) < 2:
            continue

        # 스택 후 intra-class correlation 계산
        feats = np.stack(feature_list)
        sim_matrix = cosine_similarity(feats)

        # 각 feature의 평균 correlation 계산
        mean_corr = np.mean(sim_matrix, axis=1)

        # 상위 tau quantile의 feature만 사용
        threshold = np.quantile(mean_corr, tau)
        high_corr_feats = feats[mean_corr >= threshold]

        # Semantic Center 계산 (평균 + 정규화)
        semantic_center = np.mean(high_corr_feats, axis=0)
        semantic_center /= np.linalg.norm(semantic_center) + 1e-8

        semantic_centers[cls_name] = semantic_center

    # npy 파일로 저장
    np.save(save_path, semantic_centers)
    print(f"✅ Saved semantic centers for {os.path.basename(layer_path)} → {save_path}")


# -----------------------------
# 모든 레이어 반복
# -----------------------------
for layer in sorted(os.listdir(BASE_DIR)):
    layer_path = os.path.join(BASE_DIR, layer)
    if not os.path.isdir(layer_path):
        continue

    save_path = os.path.join(SAVE_DIR, f"{layer}_centers.npy")
    build_semantic_center_for_layer(layer_path, save_path, tau=0.7)

print("\n🎯 All layers processed successfully.")