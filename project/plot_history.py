import json
from pathlib import Path

import matplotlib.pyplot as plt


HISTORY_PATH = Path("results/training_history.json")
OUTPUT_PATH = Path("results/training_curves.png")


def load_training_history(history_path: Path) -> dict:
    """Load saved training and validation metrics."""

    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history was not found: {history_path}"
        )

    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    required_keys = {
        "train_losses",
        "val_losses",
        "train_accuracies",
        "val_accuracies",
    }

    missing_keys = required_keys.difference(history)

    if missing_keys:
        raise KeyError(
            f"Training history is missing keys: {sorted(missing_keys)}"
        )

    return history


def plot_training_curves(
    history: dict,
    output_path: Path,
) -> None:
    """Plot and save loss and accuracy curves."""

    train_losses = history["train_losses"]
    val_losses = history["val_losses"]
    train_accuracies = history["train_accuracies"]
    val_accuracies = history["val_accuracies"]

    epochs = range(1, len(train_losses) + 1)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].plot(
        epochs,
        train_losses,
        marker="o",
        label="Training Loss",
    )
    axes[0].plot(
        epochs,
        val_losses,
        marker="o",
        label="Validation Loss",
    )
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_xticks(list(epochs))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs,
        train_accuracies,
        marker="o",
        label="Training Accuracy",
    )
    axes[1].plot(
        epochs,
        val_accuracies,
        marker="o",
        label="Validation Accuracy",
    )
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xticks(list(epochs))
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Training curves saved: {output_path}")


def main() -> None:
    history = load_training_history(HISTORY_PATH)

    plot_training_curves(
        history=history,
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()