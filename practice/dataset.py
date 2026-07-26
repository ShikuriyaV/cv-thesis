import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def create_dataloaders(
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create CIFAR-10 training, validation, and test data loaders."""

    # Convert images to PyTorch tensors.
    transform = transforms.ToTensor()

    # CIFAR-10 provides 50,000 training images.
    full_train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    # Split the original training set into:
    # 45,000 training images and 5,000 validation images.
    train_size = 45_000
    val_size = 5_000

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # CIFAR-10 provides a separate test set with 10,000 images.
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def main() -> None:
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=64
    )

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Training batch images:", images.shape)
    print("Training batch labels:", labels.shape)


if __name__ == "__main__":
    main()