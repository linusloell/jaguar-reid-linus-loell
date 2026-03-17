# Jaguar ReID Configuration Reference

## Purpose

This repository test different configurations for a classification model on images of jaguar from the Pantanal National park. The challenge was presented in two Kaggle competitions: [Jaguar Re-Id](https://www.kaggle.com/competitions/jaguar-re-id) and [Round 2: Jaguar Re-Id](https://www.kaggle.com/competitions/round-2-jaguar-reidentification-challenge/)

## Experiments

All experiments conducted are documented in [EDA_experiments.md](EDA_experiments.md) and [LEADERBOARD_EXPERIMENTS.md](LEADERBOARD_EXPERIMENTS.md)

The runs where logged to [Weights and Biases](https://wandb.ai/linus-loell/jaguar-reid-linus-loell/) and an overview of all experiment data can be found in the [wandb report](https://wandb.ai/linus-loell/jaguar-reid-linus-loell/reports/Reports--VmlldzoxNjIzMzg0Ng)

## Notebook

This project loads a run configuration from JSON files.

Set a `RUN_CONFIG_PATH` environment variable to select the config file.

## Quick Start

Provide a config:

```bash
export RUN_CONFIG_PATH=../configs/my_experiment.json
```

Notes:

- Path-like fields (`data_dir`, `checkpoint_dir`) are loaded as strings from JSON and converted to `Path` objects in the notebook code.
- If `seed` is present in config, it is applied to NumPy and PyTorch for reproducibility.

## Config Keys

Current default file: `configs/baseline.json`

| Key                                | Explanation                                                       | Options seen in `configs/*.json`                                                                                                                                                                                          |
| ---------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_name`                         | Weights & Biases run name.                                        | Free text                                                                                                                                                                                                                 |
| `challenge`                        | Dataset/challenge selection used by the notebook logic.           | `"challenge1"`, `"challenge2"`                                                                                                                                                                                            |
| `data_dir`                         | Root data directory containing csv files and image folders.       | `"data/challenge1"`, `"data/challenge2"`                                                                                                                                                                                  |
| `checkpoint_dir`                   | Directory where checkpoints, plots, and submissions are written.  | `"checkpoints"`                                                                                                                                                                                                           |
| `model`                            | Backbone model identifier (Hugging Face / timm model name).       | `"hf-hub:BVRA/MegaDescriptor-L-384"`, `"facebook/dinov3-vitl16-pretrain-lvd1689m"`, `"facebook/convnextv2-tiny-22k-224"`, `"google/efficientnet-b7"`, `"microsoft/resnet-18"`, `"microsoft/swin-tiny-patch4-window7-224"` |
| `model_family`                     | Optional backbone family hint used in model setup code.           | `"dinov3"`, `"convnextv2"`, `"efficientnet"`, `"resnet"`, `"swintransformer"`                                                                                                                                             |
| `input_size`                       | Input image resolution used by transforms and forward pass setup. | `244`, `384`, `386`                                                                                                                                                                                                       |
| `embedding_dim`                    | Output dimension of the projection head.                          | `256`                                                                                                                                                                                                                     |
| `hidden_dim`                       | Hidden dimension inside the projection MLP.                       | `512`                                                                                                                                                                                                                     |
| `loss_type`                        | Selects the loss/head variant.                                    | `"arcface"`, `"focal"`, `"step"`                                                                                                                                                                                          |
| `focal_alpha`                      | Alpha parameter for focal loss.                                   | `0.25`                                                                                                                                                                                                                    |
| `focal_gamma`                      | Gamma parameter for focal loss.                                   | `2.0`                                                                                                                                                                                                                     |
| `focal_reduction`                  | Reduction mode for focal loss.                                    | `"mean"`, `"ce"`                                                                                                                                                                                                          |
| `arcface_margin`                   | Angular margin for ArcFace head (radians).                        | `0.5`                                                                                                                                                                                                                     |
| `arcface_scale`                    | ArcFace logits scale.                                             | `64.0`                                                                                                                                                                                                                    |
| `sphere_m`                         | Angular multiplier for SphereFace (`cos(m*theta)`).               | `4`                                                                                                                                                                                                                       |
| `sphere_scale`                     | SphereFace logits scale.                                          | `64.0`                                                                                                                                                                                                                    |
| `dropout`                          | Dropout probability in projection head.                           | `0.3`                                                                                                                                                                                                                     |
| `batch_size`                       | Training/embedding dataloader batch size.                         | `32`                                                                                                                                                                                                                      |
| `learning_rate`                    | Base learning rate for optimizer setup.                           | `1e-4`, `3e-4`                                                                                                                                                                                                            |
| `optimizer_type`                   | Optimizer selection.                                              | `"muon"`, `"sgd"`                                                                                                                                                                                                         |
| `optim_muon_learning_rate`         | Muon-specific learning rate.                                      | `0.02`                                                                                                                                                                                                                    |
| `optim_muon_momentum`              | Muon momentum.                                                    | `0.95`                                                                                                                                                                                                                    |
| `optim_sgd_momentum`               | SGD momentum.                                                     | `0.9`                                                                                                                                                                                                                     |
| `lr_scheduler_type`                | Learning rate scheduler strategy.                                 | `"reduce_on_plateau"`, `"cosine"`, `"cosine_warmup"`, `"step"`, `"exponential"`                                                                                                                                           |
| `lr_scheduler_factor`              | Reduce-on-plateau multiplicative decay factor.                    | `0.5`                                                                                                                                                                                                                     |
| `lr_scheduler_patience`            | Reduce-on-plateau patience (epochs).                              | `5`                                                                                                                                                                                                                       |
| `lr_scheduler_T_max`               | Cosine/Cosine-warmup cycle length for cosine phase.               | `45`, `50`                                                                                                                                                                                                                |
| `lr_scheduler_eta_min`             | Minimum learning rate for cosine-family schedulers.               | `5e-7`, `1e-6`                                                                                                                                                                                                            |
| `lr_scheduler_warmup_epochs`       | Number of warmup epochs before cosine phase.                      | `5`                                                                                                                                                                                                                       |
| `lr_scheduler_warmup_start_factor` | Initial warmup LR multiplier.                                     | `0.1`                                                                                                                                                                                                                     |
| `lr_scheduler_step_size`           | Step scheduler decay interval (epochs).                           | `15`                                                                                                                                                                                                                      |
| `lr_scheduler_gamma`               | Multiplicative gamma for step/exponential schedulers.             | `0.1`, `0.9`                                                                                                                                                                                                              |
| `weight_decay`                     | Weight decay regularization coefficient.                          | `1e-4`                                                                                                                                                                                                                    |
| `num_epochs`                       | Maximum training epochs.                                          | `50`, `70`                                                                                                                                                                                                                |
| `patience`                         | Early stopping patience on validation loss.                       | `10`                                                                                                                                                                                                                      |
| `val_split`                        | Fraction of training data used for validation.                    | `0.2`                                                                                                                                                                                                                     |
| `seed`                             | Random seed for split and reproducibility.                        | `42`                                                                                                                                                                                                                      |
| `train_bg_intervention`            | Background intervention applied to training images.               | `"none"`, `"noise"`                                                                                                                                                                                                       |
| `val_bg_intervention`              | Background intervention applied to validation images.             | `"none"`, `"noise"`                                                                                                                                                                                                       |
