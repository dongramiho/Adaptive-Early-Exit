import os
import torch
import torchvision.models as models
import torchvision.transforms as T
from torchvision.models import ResNet101_Weights
from torch.utils.data import DataLoader
from utils.dataset_loader import UCF101Frames
from utils.layer_hook import FeatureExtractor
from utils.save_features import save_feature

# -----------------------------------------------------------
# ✅ 1. Model Load (pretrained on ImageNet)
# -----------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet101_Weights.IMAGENET1K_V2
model = models.resnet101(weights=weights).to(device)
model.eval()

# Register hooks for intermediate layers
layers_to_extract = ["layer1", "layer2", "layer3", "layer4"]
extractor = FeatureExtractor(model, layers_to_extract)

# -----------------------------------------------------------
# ✅ 2. Transform (ImageNet normalization)
# -----------------------------------------------------------
transform = weights.transforms()

# -----------------------------------------------------------
# ✅ 3. Dataset & Loader (UCF101 Frames)
# -----------------------------------------------------------
data_root = "/home/ubuntu/lab_ws/R3D/mmaction2/tools/data/ucf101/videos"
dataset = UCF101Frames(root=data_root, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# -----------------------------------------------------------
# ✅ 4. Feature Extraction (COACH-style temporal averaging)
# -----------------------------------------------------------
save_dir = "./output"
os.makedirs(save_dir, exist_ok=True)

with torch.no_grad():
    for idx, (frames, labels, paths) in enumerate(loader):
        frames = frames.to(device)

        # ------------------------------
        # 🔹 [추가된 부분] Temporal averaging (COACH 반영)
        # ------------------------------
        # UCF101 비디오에서 중앙 5프레임(mean pooling)
        if frames.ndim == 5:  # (B, T, C, H, W)
            mid = frames.shape[1] // 2
            frames = frames[:, mid-2:mid+3]  # 중앙 5프레임 추출
            frames = frames.mean(dim=1)       # 시간축 평균 (temporal mean)
        elif frames.ndim == 4:
            # 단일 프레임인 경우 그대로 사용
            pass

        # 모델 forward
        _ = model(frames)
        features = extractor.get_features()

        # ------------------------------
        # 🔹 Save layer features
        # ------------------------------
        for layer_name, feat in features.items():
            save_feature(feat.cpu(), save_dir, layer_name, paths[0])

        if idx % 100 == 0:
            print(f"[{idx}/{len(loader)}] processed: {paths[0]}")

print("\n✅ Feature extraction (with temporal averaging) completed successfully.")