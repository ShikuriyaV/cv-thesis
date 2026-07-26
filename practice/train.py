from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import create_dataloaders
from model import SimpleCNN


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if (batch_index + 1) % 100 == 0:
            print(
                f"Batch {batch_index + 1}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / total_samples


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on the validation set."""

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
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

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=64
    )

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=0.001,
    )

    num_epochs = 5
    best_val_accuracy = 0.0

    # Save the model beside this train.py file.
    model_path = Path(__file__).parent / "best_model.pth"

    print("Device:", device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        training_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(f"Training Loss: {training_loss:.4f}")
        print(f"Validation Loss: {validation_loss:.4f}")
        print(
            f"Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )

        if validation_accuracy > best_val_accuracy:
            best_val_accuracy = validation_accuracy

            torch.save(
                model.state_dict(),
                model_path,
            )

            print(
                f"Best model saved: "
                f"{best_val_accuracy * 100:.2f}%"
            )

    print("\nTraining finished.")
    print(f"Best model path: {model_path}")
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    main()