# Jaguar ReID Configuration Reference

This project loads run configuration from JSON files.

Config resolution order:
1. If environment variable `RUN_CONFIG_PATH` is set, that file is used.
2. Otherwise, the default is `../configs/baseline.json`.

You can create multiple JSON config files in `configs/` and switch runs by pointing `RUN_CONFIG_PATH` to the desired file.

## Quick Start

Provide a config:

```bash
export RUN_CONFIG_PATH=../configs/my_experiment.json
```

Notes:
- Path-like fields (`data_dir`, `checkpoint_dir`) are loaded as strings from JSON and converted to `Path` objects in the notebook code.
- If `seed` is present in config, it is applied to NumPy and PyTorch for reproducibility.

## Baseline Config Keys

Current default file: `configs/baseline.json`

| Key | Explanation | options |
|---|---|---|
| `run_name` | wandb run name |  |
| `data_dir` | Root data directory containing `train.csv`, `test.csv`, image folders, etc. |  |
| `checkpoint_dir` | Directory where checkpoints, plots, and submission files are written. |  |
| `model` | Backbone model identifier passed to `timm.create_model`. | TODO |
| `input_size` | Input image resolution used by preprocessing and dummy forward pass. |  |
| `embedding_dim` | Output dimension of the projection head. |  |
| `hidden_dim` | Hidden dimension inside the projection MLP. | |
| `loss_type` | Selects the classification/margin-loss head and training loss setup. | `"arcface"`, `"ce"`, `"focal"`, `"sphere"`. |
| `focal_alpha` | Alpha parameter for focal loss class balancing. |  |
| `focal_gamma` | Gamma parameter for focal loss focusing strength. |  |
| `focal_reduction` | Reduction mode for `torchvision.ops.sigmoid_focal_loss`. |  `"none"`, `"mean"`, `"sum"`. |
| `arcface_margin` | Angular margin (radians) used by ArcFace head. | |
| `arcface_scale` | Logit scale factor for ArcFace head. |  |
| `sphere_m` | Angular multiplier `m` used by SphereFace head (`cos(m*theta)`). |  |
| `sphere_scale` | Logit scale factor for SphereFace head. | |
| `dropout` | Dropout probability in the projection head. |  |
| `batch_size` | Batch size for embedding extraction and training dataloaders. |  |
| `learning_rate` | Base optimizer learning rate (`AdamW`). |  |
| `lr_scheduler_type` | Learning rate scheduler strategy. | `"reduce_on_plateau"`, `"exponential"`, `"cosine"`, `"cosine_warmup"`, `"step"`. |
| `weight_decay` | Weight decay used by `AdamW`. |  |
| `num_epochs` | Maximum number of training epochs. | |
| `patience` | Early stopping patience (epochs without val-loss improvement). |  |
| `val_split` | Fraction of training data used for validation split. |  |
| `seed` | Random seed used for reproducibility (NumPy/PyTorch and split functions). |  |
