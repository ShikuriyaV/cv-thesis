import argparse
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

import json

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from pathlib import Path

import torch

from project.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TARGET_CLASSES,
    build_dataloaders,
)

from project.utils import build_model, get_device, load_model


DEFAULT_CHECKPOINT_PATH = Path(
    "project/checkpoints/best_model.pth"
)

REPORT_PATH = Path(
    "results/classification_report.txt"
)

CONFUSION_DATA_PATH = Path(
    "results/confusion_matrix.json"
)

SAMPLE_OUTPUT_PATH = Path(
    "results/sample_predictions.png"
)

def load_training_history(history_path: Path) -> dict:
    """Load the saved training and validation metrics."""

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
    """Plot and save the loss and accuracy curves."""

    train_losses = history["train_losses"]
    val_losses = history["val_losses"]
    train_accuracies = history["train_accuracies"]
    val_accuracies = history["val_accuracies"]

    num_epochs = len(train_losses)
    epochs = range(1, num_epochs + 1)

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

def collect_predictions(
    model,
    data_loader,
    device,
) -> tuple[
    list[int],
    list[int],
    list[tuple[torch.Tensor, int, int]],
    list[tuple[torch.Tensor, int, int]],
]:
    """Collect labels and example correct/incorrect predictions."""

    model.eval()

    true_labels = []
    predicted_labels = []

    correct_samples = []
    incorrect_samples = []

    with torch.no_grad():
        for images, labels in data_loader:
            images_on_device = images.to(device)

            outputs = model(images_on_device)
            predictions = outputs.argmax(dim=1).cpu()

            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.tolist())

            for image, true_label, predicted_label in zip(
                images,
                labels,
                predictions,
            ):
                sample = (
                    image.clone(),
                    int(true_label),
                    int(predicted_label),
                )

                if (
                    true_label == predicted_label
                    and len(correct_samples) < 5
                ):
                    correct_samples.append(sample)

                elif (
                    true_label != predicted_label
                    and len(incorrect_samples) < 5
                ):
                    incorrect_samples.append(sample)

    return (
        true_labels,
        predicted_labels,
        correct_samples,
        incorrect_samples,
    )

def calculate_and_save_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    report_path: Path,
    confusion_data_path: Path,
) -> None:
    """Calculate, print, and save classification metrics."""

    class_labels = list(range(len(class_names)))

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=class_labels,
            average=None,
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_labels,
    )

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    print("\nPer-class metrics:")

    for index, class_name in enumerate(class_names):
        print(
            f"{class_name}: "
            f"Precision={precision[index]:.4f}, "
            f"Recall={recall[index]:.4f}, "
            f"F1={f1[index]:.4f}, "
            f"Support={support[index]}"
        )

    print("\nClassification Report:")
    print(report)

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_content = (
        f"Test Accuracy: {accuracy:.4f}\n"
        f"Macro-F1: {macro_f1:.4f}\n"
        f"Weighted-F1: {weighted_f1:.4f}\n\n"
        f"{report}"
    )

    report_path.write_text(
        report_content,
        encoding="utf-8",
    )

    print(
        f"Classification report saved: {report_path}"
    )
    
    confusion_data = {
        "class_names": class_names,
        "matrix": matrix.tolist(),
    }

    confusion_data_path.write_text(
        json.dumps(
            confusion_data,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        f"Confusion matrix data saved: "
        f"{confusion_data_path}"
    )

def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization."""

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=image.dtype,
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        dtype=image.dtype,
    ).view(3, 1, 1)

    image = image * std + mean

    return image.clamp(0, 1)

def save_sample_predictions(
    correct_samples,
    incorrect_samples,
    class_names,
    output_path: Path,
) -> None:
    """Save five correct and five incorrect predictions."""

    samples = [
        ("Correct", sample)
        for sample in correct_samples
    ] + [
        ("Incorrect", sample)
        for sample in incorrect_samples
    ]

    if not samples:
        raise ValueError("No prediction samples were collected.")

    image_size = 224
    text_height = 55
    columns = 5
    rows = 2

    canvas = Image.new(
        "RGB",
        (
            columns * image_size,
            rows * (image_size + text_height),
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    for index, (result_type, sample) in enumerate(samples):
        image, true_label, predicted_label = sample

        image = denormalize_image(image)
        pil_image = to_pil_image(image)

        row = index // columns
        column = index % columns

        x = column * image_size
        y = row * (image_size + text_height)

        canvas.paste(pil_image, (x, y))

        text = (
            f"{result_type}\n"
            f"True: {class_names[true_label]}\n"
            f"Pred: {class_names[predicted_label]}"
        )

        draw.text(
            (x + 5, y + image_size + 2),
            text,
            fill="black",
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(output_path)

    print(f"Sample predictions saved: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ResNet18 model on the test dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the trained model checkpoint.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {args.checkpoint}"
        )

    device = get_device()

    _, _, test_loader = build_dataloaders(
        batch_size=32,
        num_workers=0,
    )

    model = build_model(
        num_classes=len(TARGET_CLASSES)
    ).to(device)

    load_model(
        model=model,
        checkpoint_path=str(args.checkpoint),
        device=device,
    )

    (
        y_true,
        y_pred,
        correct_samples,
        incorrect_samples,
    ) = collect_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    print("Device:", device)
    print("Classes:", TARGET_CLASSES)
    print("Test samples:", len(y_true))
    print("Predictions collected:", len(y_pred))
    print("First 10 true labels:", y_true[:10])
    print("First 10 predictions:", y_pred[:10])

    calculate_and_save_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=TARGET_CLASSES,
        report_path=REPORT_PATH,
        confusion_data_path=CONFUSION_DATA_PATH,
    )

    save_sample_predictions(
        correct_samples=correct_samples,
        incorrect_samples=incorrect_samples,
        class_names=TARGET_CLASSES,
        output_path=SAMPLE_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()