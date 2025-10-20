from torchvision.io import read_video
from torchvision import transforms
import os
from torch.utils.data import Dataset
import torch
from PIL import Image

class UCF101Frames(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.video_paths = []

        # 각 클래스별 비디오 경로 수집
        for class_name in sorted(os.listdir(root)):
            class_dir = os.path.join(root, class_name)
            if os.path.isdir(class_dir):
                for vid in os.listdir(class_dir):
                    if vid.endswith(".avi"):
                        self.video_paths.append(os.path.join(class_dir, vid))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        try:
            # PyTorch의 read_video는 (T, H, W, C) 형식 반환
            frames, _, _ = read_video(video_path, pts_unit='sec')
        except Exception as e:
            print(f"[WARN] Error reading {video_path}: {e}")
            # 비디오 읽기 실패 시 dummy tensor 반환
            dummy = torch.zeros((1, 3, 224, 224))
            return dummy, 0, video_path

        if len(frames) == 0:
            print(f"[WARN] No frames in video: {video_path}")
            dummy = torch.zeros((1, 3, 224, 224))
            return dummy, 0, video_path

        # 중앙 프레임 선택
        mid = len(frames) // 2
        frame = frames[mid]  # (H, W, C)

        # 안전하게 RGB 변환
        if frame.shape[-1] > 3:
            frame = frame[:, :, :3]

        # (H, W, C) → (C, H, W)
        frame = frame.permute(2, 0, 1)

        # PIL 이미지로 변환
        img = transforms.ToPILImage()(frame)

        # transform 적용 (ImageNet normalization 등)
        if self.transform:
            img = self.transform(img)

        # (1, C, H, W) 형태로 반환
        return img, 0, video_path  # 배치 차원 제거
        