DinoV3 with RoPE (Rotary Positional Embeddings)?

# EDA Experiments

## Loss Functions

ArcFace, ArcCos, Focal Loss, and Cross Entropy

Validity requirements:
- Controlled comparison (same backbone, schedule, augmentations, embedding dimension, evaluation)
- Report identity-balanced mAP and training stability notes
- Interpretation of why the better loss fits this dataset

Apply only if the experiment meets the Valid (1.0) bar.
- 2 loss functions: 1.00
- 3 loss functions: 1.50
- 4 loss functions: 2.00
- 5 loss functions: 2.50
- 6 loss functions: 3.00
- 7 loss functions: 3.50
- 8 loss functions: 4.00
Cap: 4.00 (8 or more losses)
