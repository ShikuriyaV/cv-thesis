from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import create_dataloaders
from model import SimpleCNN


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the trained model on the test set."""

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Only the test loader is needed here.
    _, _, test_loader = create_dataloaders(batch_size=64)

    # Recreate the same model structure.
    model = SimpleCNN().to(device)

    model_path = Path(__file__).parent / "best_model.pth"

    # Load the saved parameters.
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("Device:", device)
    print("Loaded model:", model_path)
    print("Test samples:", len(test_loader.dataset))
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()