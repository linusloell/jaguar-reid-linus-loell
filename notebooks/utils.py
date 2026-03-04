from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Data loading & preview
# ---------------------------------------------------------------------------

def load_train_data(data_dir: Path, subset_fraction: float | None = None, seed: int = 42) -> pd.DataFrame:
    """Load train.csv and print a basic summary.

    Args:
        data_dir:         Root directory that contains ``train.csv``.
        subset_fraction:  If given (e.g. ``0.1`` for 10 %), a stratified random
                          sample of that fraction is returned. Must be in (0, 1).
                          ``None`` returns the full dataset.
        seed:             Random seed used when sampling a subset.

    Returns:
        The (optionally sub-sampled) training DataFrame with columns
        ``filename`` and ``ground_truth``.
    """
    train_df = pd.read_csv(data_dir / "train.csv")

    if subset_fraction is not None:
        if not 0 < subset_fraction < 1:
            raise ValueError(f"subset_fraction must be in (0, 1), got {subset_fraction}")
        train_df = train_df.groupby("ground_truth", group_keys=False).apply(
            lambda g: g.sample(frac=subset_fraction, random_state=seed)
        ).reset_index(drop=True)
        print(f"Using {subset_fraction:.0%} stratified subset of the data")

    print("Training dataset:")
    print(f"  Total images:      {len(train_df)}")
    print(f"  Unique identities: {train_df['ground_truth'].nunique()}")
    print("\nSample rows:")
    print(train_df.head())

    identity_counts = train_df["ground_truth"].value_counts()
    print("\nIdentity distribution:")
    print(f"  Min images per identity: {identity_counts.min()} ({identity_counts.idxmin()})")
    print(f"  Max images per identity: {identity_counts.max()} ({identity_counts.idxmax()})")
    print(f"  Mean images per identity: {identity_counts.mean():.1f}")

    return train_df


def plot_identity_distribution(train_df: pd.DataFrame, log_wandb: bool = True) -> None:
    """Plot a bar chart of images-per-identity and optionally log it to W&B.

    Args:
        train_df:   DataFrame returned by :func:`load_train_data`.
        log_wandb:  When ``True`` the figure is logged to the active W&B run.
    """
    identity_counts = train_df["ground_truth"].value_counts()

    fig, ax = plt.subplots(figsize=(14, 5))
    identity_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xlabel("Jaguar Identity")
    ax.set_ylabel("Number of Images")
    ax.set_title("Training Data: Images per Jaguar Identity")
    ax.axhline(
        y=identity_counts.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {identity_counts.mean():.1f}",
    )
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if log_wandb:
        wandb.log({"identity_distribution_full": wandb.Image(fig)})

    plt.show()

    min_samples_for_split = 2
    low_sample_identities = identity_counts[identity_counts < min_samples_for_split]
    if len(low_sample_identities) > 0:
        print(
            f"\nWarning: {len(low_sample_identities)} identities have fewer than "
            f"{min_samples_for_split} images"
        )


def create_train_val_split(
    train_df: pd.DataFrame,
    val_split: float = 0.2,
    seed: int = 42,
    log_wandb: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder, int]:
    """Create a stratified train/validation split and log distributions to W&B.

    Every jaguar identity is guaranteed to appear in both the training and
    validation sets.

    Args:
        train_df:   DataFrame returned by :func:`load_train_data`.
        val_split:  Fraction of data to reserve for validation.
        seed:       Random seed for reproducibility.
        log_wandb:  When ``True`` distributions are logged to the active W&B run.

    Returns:
        A 4-tuple of ``(train_data, val_data, label_encoder, num_classes)``.
    """
    label_encoder = LabelEncoder()
    train_df = train_df.copy()
    train_df["label_encoded"] = label_encoder.fit_transform(train_df["ground_truth"])
    num_classes = len(label_encoder.classes_)

    train_data, val_data = train_test_split(
        train_df,
        test_size=val_split,
        random_state=seed,
        stratify=train_df["ground_truth"],
    )

    print("Dataset split:")
    print(f"  Training:   {len(train_data)} images ({100 * (1 - val_split):.0f}%)")
    print(f"  Validation: {len(val_data)} images ({100 * val_split:.0f}%)")

    train_identities = set(train_data["ground_truth"].unique())
    val_identities = set(val_data["ground_truth"].unique())

    print("\nIdentity coverage:")
    print(f"  Identities in training:   {len(train_identities)}")
    print(f"  Identities in validation: {len(val_identities)}")
    print(f"  Overlap: {len(train_identities & val_identities)}")
    if train_identities == val_identities:
        print("  ✓ All identities present in both sets")

    train_counts = train_data["ground_truth"].value_counts().sort_index()
    val_counts = val_data["ground_truth"].value_counts().sort_index()

    if log_wandb:
        distribution_df = pd.DataFrame(
            {
                "identity": train_counts.index,
                "train_count": train_counts.values,
                "val_count": val_counts.values,
                "total_count": train_counts.values + val_counts.values,
                "train_ratio": train_counts.values
                / (train_counts.values + val_counts.values),
            }
        )

        wandb.log(
            {
                "identity_distribution_table": wandb.Table(dataframe=distribution_df),
                "num_identities": num_classes,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "train_samples_per_identity": wandb.Histogram(train_counts.values),
                "val_samples_per_identity": wandb.Histogram(val_counts.values),
            }
        )

    # Side-by-side bar chart
    fig, ax = plt.subplots(figsize=(14, 5))
    width = 0.35
    x = np.arange(len(train_counts))
    ax.bar(x - width / 2, train_counts.values, width, label="Train", color="steelblue")
    ax.bar(x + width / 2, val_counts.values, width, label="Validation", color="coral")
    ax.set_xlabel("Jaguar Identity")
    ax.set_ylabel("Number of Images")
    ax.set_title("Train vs Validation: Images per Identity")
    ax.set_xticks(x)
    ax.set_xticklabels(train_counts.index, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()

    if log_wandb:
        wandb.log({"train_val_distribution": wandb.Image(fig)})

    plt.show()

    print(f"\nLogged identity distributions to W&B")
    print(
        f"  Train samples per identity: {train_counts.min()} – {train_counts.max()} "
        f"(mean: {train_counts.mean():.1f})"
    )
    print(
        f"  Val samples per identity:   {val_counts.min()} – {val_counts.max()} "
        f"(mean: {val_counts.mean():.1f})"
    )

    return train_data, val_data, label_encoder, num_classes
