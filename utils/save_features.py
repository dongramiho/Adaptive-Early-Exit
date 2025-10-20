import os
import numpy as np

def save_feature(feat_tensor, base_dir, layer_name, path):
    cls = path.split("/")[-2]
    vid = os.path.basename(path).replace(".avi", "")
    out_dir = os.path.join(base_dir, layer_name, cls)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{vid}.npy"), feat_tensor.numpy())