# Adaptive-Early-Exit  
## Temporal-aware Early-Exit for Collaborative Inference  

This repository implements an **early-exit calibration framework** that combines  
**semantic separability (ΔS)** and a **temporal consistency indicator**,  
inspired by *COACH: Near Bubble-free Pipeline Collaborative Inference* (INFOCOM 2025).

---

## Overview  

Recent research such as **COACH (INFOCOM 2025)** has shown that collaborative inference pipelines  
can greatly benefit from temporally stable intermediate features when determining semantic separability.  
However, in shallow or mid-level layers of DNNs (e.g., **ResNet-101 layer3**),  
semantic centers tend to be weakly discriminative.  

To address this, we introduce a **temporal consistency auxiliary metric**  
that captures the similarity between consecutive frame features,  
stabilizing the ΔS distribution and enabling more reliable early-exit decisions.

---

## Methodology  

### 1. Feature Extraction  
- **Backbone:** ResNet-101 (torchvision.models, pretrained on ImageNet1K_V2)  
- **Dataset:** UCF101 (video action recognition benchmark)  
- Extracted layer1–layer4 features using PyTorch forward hooks.  
- Applied temporal mean pooling (central 5 frames per video):  

```python
frames = frames[:, mid-2:mid+3]  # select central 5 frames
frames = frames.mean(dim=1)      # temporal averaging
```

→ This temporal averaging follows the COACH framework for stable feature representation.  

---

### 2. Semantic Center Construction  

For each class `c`:

1. Compute feature correlations within the same class.  
2. Select top-τ (e.g., 70%) highly correlated features.  
3. Compute normalized class semantic center:  

```python
semantic_center_c = mean(selected_features) / ||mean(selected_features)||
```

---

### 3. Early-Exit Threshold Calibration  

- Measure semantic separability between top-1 and top-2 similarity scores:  

ΔS = (S_top1 - S_top2) / (|S_top1| + |S_top2| + ε)

- Combine with temporal consistency (frame-wise cosine similarity):  

Hybrid = α * ΔS + β * s_temp  

where:  
s_temp = clip(dot(f_t-1, f_t), 0, 1)  

Typical values: α = 0.9, β = 0.1  

- Find minimum τ satisfying target edge precision (e.g., 95%):  

```python
if precision(ΔS >= τ) >= 0.95:
    best_tau = τ
```

---

### 4. Evaluation (Simulation)  

- **Layer:** layer3 (ResNet-101)  
- **Metric:** Exit Rate vs Edge Exit Accuracy  

Example result:  
```
Calibrated τ_exit = 0.102
Exit Rate = 20.39%
Edge Exit Accuracy = 95.36%
```

---

## Key Insight  

The proposed temporal consistency auxiliary metric enables early-exit  
even at semantically ambiguous intermediate layers by stabilizing inter-frame  
feature variations — effectively mimicking the temporal mean feature strategy of COACH.  

This makes early-exit decision-making robust in dynamic scenes  
while maintaining minimal computational overhead (<0.001% of total FLOPs).

---

## Reference  

### Backbone & Dataset  
- **ResNet-101** — He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016  
- **UCF101** — Soomro et al., *UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild*, CRCV 2012  

### Base Framework  
- **COACH** — Zhang et al., *COACH: Near Bubble-free Pipeline Collaborative Inference*, IEEE INFOCOM 2025  
  (Referenced for temporal mean pooling and semantic separability calibration)



---
