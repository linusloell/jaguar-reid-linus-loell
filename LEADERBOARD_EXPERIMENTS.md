# Leaderboard Experiments

## 1. Which Backbone is best?

Foundations:

- Megadescriptor
- DINOv3
- ConvNeXt v2: facebook/convnextv2-tiny-22k-224
- SwinTransformers

Lightweight:

- NFNet
- EfficientNet
- MobileNet v3
- ResNet18

### report:

Why these backbones
Table: mAP and efficiency metrics
Interpretation: what characteristics matter and why

### Results

| Model Name | HF Path | Parameters | MAP |
|---|---|---|---|
| Megadescriptor | BVRA/MegaDescriptor-L-384 | 195,198,516 | 0.7723 |
| DINOv3 | facebook/dinov3-vitl16-pretrain-lvd1689m | 303,129,600 | 0.8923 |
| ConvNeXt v2 | facebook/convnextv2-tiny-22k-224 | 27,866,496 | 0.7302 |
| Swin Transformer | microsoft/swin-tiny-patch4-window7-224 | 27,519,354 | 0.7507 |
| EfficientNet | google/efficientnet-b7 | 63,786,960 | 0.8329 |
| ResNet18 | microsoft/resnet-18 | 11,176,512 | 0.6531 |


# wich optimizer is best?

# wich LR-scheduler is best?
- reduce on plateu (baseline)
- exponential
- cosine annealing
- step decay