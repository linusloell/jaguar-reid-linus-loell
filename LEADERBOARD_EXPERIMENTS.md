# Leaderboard Experiments

## Requirements
**Q5:** I compared multiple backbones (for example, ResNet18 vs DINOv3 vs EfficientNet vs MegaDescriptor). How is this scored?
Ruling: One experiment. Score depends on control and analysis, plus backbone-count bonus.
Where to document this: LEADERBOARD_EXPERIMENTS.md (optional deeper diagnostics in EDA with cross-reference)

Base requirements:
Same training protocol, loss, schedule, augmentation, evaluation
Same embedding dimension (or justification)
Report mAP and at least one efficiency metric

Scoring for backbone comparison:
Base score: 1.0 if Valid criteria are met
Bonus: +0.20 per backbone included in the controlled comparison
2 backbones: 1.20
3 backbones: 1.40
4 backbones: 1.60
5 or more backbones: 2.00 (cap)

What to document:
Why these backbones
Table: mAP and efficiency metrics
Interpretation: what characteristics matter and why

**Q23:** I compared different optimizers (Adam, AdamW, Muon, SGD with momentum) and learning rate schedulers (cosine annealing, one cycle policy, reduce-on-plateau). Does this count as a valid experiment?
Ruling: 1.0 (Valid experiment)

Where to document this:
If the goal is understanding training stability and sensitivity: EDA_EXPERIMENTS.md
If the goal is improving the public leaderboard and it changes the final pipeline: LEADERBOARD_EXPERIMENTS.md
Choose one as primary and cross-reference the other if needed.
Rationale: Yes. Optimizer choice and learning rate scheduling are core training design decisions. A controlled comparison answers a clear research question, such as “Which optimizer or scheduler yields the best identity-balanced mAP and stability for this dataset and model?”

How to count experiments:
Comparing multiple optimizers under one fixed scheduler is one experiment (“Which optimizer works best?”).
Comparing multiple schedulers under one fixed optimizer is one experiment (“Which scheduler works best?”).
Comparing optimizer–scheduler pairs as a grid is one experiment if framed as one question (“Which combination works best?”), but it must be documented as a structured study, not ad hoc tuning.

Validity requirements (minimum):
Controlled setup: same backbone, loss, augmentations, batch size, embedding dimension, training length, and evaluation protocol
Clear definitions: optimizer hyperparameters (weight decay, betas, momentum) and scheduler settings (warmup, max LR, cycle length, patience)

Report identity-balanced mAP plus training stability indicators (divergence rate, variance across seeds, convergence curves)
What to document:
The comparison plan (which optimizers, which schedulers, and why)

A results table with identity-balanced mAP for each condition
Training dynamics: convergence speed, stability, and sensitivity
Mean and standard deviation across seeds for top contenders
Interpretation: why the best choice fits this task (regularization, noisy gradients, batch size effects)


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