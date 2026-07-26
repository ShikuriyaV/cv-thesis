import json
from pathlib import Path

import torch
from torch import nn

from project.dataset import TARGET_CLASSES, build_dataloaders
from project.utils import (
    build_model,
    get_device,
    load_model,
    save_model,
)

def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    """Train the model for one epoch."""

    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Remove gradients left from the previous batch
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        outputs = model(images)

        # Calculate classification loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update trainable parameters
        optimizer.step()

        # Calculate batch statistics
        batch_size = labels.size(0)
        predictions = outputs.argmax(dim=1)

        running_loss += loss.item() * batch_size
        running_correct += (
            predictions == labels
        ).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_correct / total_samples

    return epoch_loss, epoch_accuracy

def evaluate(
    model,
    data_loader,
    criterion,
    device,
):
    """Evaluate the model without updating its parameters."""

    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            predictions = outputs.argmax(dim=1)

            running_loss += loss.item() * batch_size
            running_correct += (
                predictions == labels
            ).sum().item()
            total_samples += batch_size

    average_loss = running_loss / total_samples
    accuracy = running_correct / total_samples

    return average_loss, accuracy

def main():
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    checkpoint_path = "project/checkpoints/best_model.pth"
    history_path = Path("results/training_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=batch_size,
        num_workers=0,
    )

    model = build_model(
        num_classes=len(TARGET_CLASSES)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=learning_rate,
    )

    print("Device:", device)
    print("Classes:", TARGET_CLASSES)
    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))
    print("Loss function:", criterion)
    print("Optimizer:", optimizer.__class__.__name__)

    best_val_accuracy = 0.0
    best_epoch = 0

    history = {
    "train_losses": [],
    "val_losses": [],
    "train_accuracies": [],
    "val_accuracies": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accuracies"].append(train_accuracy)
        history["val_accuracies"].append(val_accuracy)

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

            save_model(
                model=model,
                checkpoint_path=checkpoint_path,
            )

            print(
                f"Best model saved: "
                f"Validation Accuracy = {best_val_accuracy:.4f}"
            )

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    print(f"Training history saved: {history_path}")

    print("\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Checkpoint: {checkpoint_path}")

    load_model(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nBest model loaded for testing.")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()