import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def main() -> None:
    model = SimpleCNN()

    # Simulate 8 CIFAR-10 images
    images = torch.randn(8, 3, 32, 32)
    outputs = model(images)

    print("Input shape:", images.shape)
    print("Output shape:", outputs.shape)


if __name__ == "__main__":
    main()