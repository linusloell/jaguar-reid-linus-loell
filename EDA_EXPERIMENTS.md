DinoV3 with RoPE (Rotary Positional Embeddings)?

# EDA Experiments

## Loss Functions

### Loss Function Comparison

Compare different metric learning and classification losses:

| Loss | Applied to | Head type | Training target | Inference |
|------|-----------|-----------|-----------------|-----------|
| **ArcFace** | Logits (normalized + margin) | ArcFaceLayer | CE on margin-adjusted logits | Normalized embeddings + cosine similarity |
| **CE** | Logits only | Linear classifier | Standard cross-entropy | Embeddings + cosine similarity |
| **Sphere loss** | Logits (angular margin) | SphereLayer | CE on sphere-adjusted logits | Normalized embeddings + cosine similarity |
| **Focal loss** | Logits only | Linear classifier | Hard example weighting | Embeddings + cosine similarity |

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
