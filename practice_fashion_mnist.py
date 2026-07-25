"""Train, validate, and test a neural network on FashionMNIST."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 3
RANDOM_SEED = 42


class FashionClassifier(nn.Module):
    """A simple fully connected neural network."""

    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),                  # [batch, 1, 28, 28] -> [batch, 784]
            nn.Linear(28 * 28, 128),       # 784 inputs -> 128 neurons
            nn.ReLU(),
            nn.Linear(128, 64),            # 128 neurons -> 64 neurons
            nn.ReLU(),
            nn.Linear(64, 10),             # 10 FashionMNIST classes
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Perform forward propagation."""
        return self.network(images)


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one complete epoch."""

    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 1. Clear gradients from the previous batch.
        optimizer.zero_grad()

        # 2. Forward propagation.
        outputs = model(images)

        # 3. Calculate the prediction error.
        loss = criterion(outputs, labels)

        # 4. Backpropagation.
        loss.backward()

        # 5. Update weights and biases.
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def evaluate(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model without updating parameters."""

    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Validation and testing do not require gradient calculation.
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


def main() -> None:
    """Run the complete machine-learning workflow."""

    torch.manual_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("FashionMNIST classification")
    print("=" * 65)
    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Convert images to tensors and normalize pixel values.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Download the original FashionMNIST training dataset.
    full_training_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    # Download the official FashionMNIST test dataset.
    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    # Split the original training set:
    # 54,000 samples for training
    # 6,000 samples for validation
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_dataset, validation_dataset = random_split(
        full_training_dataset,
        [54_000, 6_000],
        generator=generator,
    )

    # num_workers=0 is the safest setting for a first Windows project.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print("\nDataset sizes")
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))
    print("Test samples:", len(test_dataset))

    model = FashionClassifier().to(device)

    # CrossEntropyLoss is commonly used for multiclass classification.
    criterion = nn.CrossEntropyLoss()

    # SGD performs gradient-descent parameter updates.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\nModel structure")
    print(model)

    print("\nStarting training")

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            device,
        )

        validation_loss, validation_accuracy = evaluate(
            validation_loader,
            model,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy * 100:.2f}% | "
            f"Validation loss: {validation_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy * 100:.2f}%"
        )

    test_loss, test_accuracy = evaluate(
        test_loader,
        model,
        criterion,
        device,
    )

    print("\nFinal test results")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    model_directory = Path("models")
    model_directory.mkdir(exist_ok=True)

    model_path = model_directory / "fashion_mnist_model.pth"

    torch.save(model.state_dict(), model_path)

    print("\nModel saved to:", model_path)
    print("Training workflow completed successfully.")


if __name__ == "__main__":
    main()