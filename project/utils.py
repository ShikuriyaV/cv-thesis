from pathlib import Path
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def get_device():
    """Automatically select GPU when CUDA is available."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    return device


def build_model(num_classes):
    """Build a ResNet18 fixed-feature-extractor model."""

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # Freeze all original ResNet18 parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the original 1000-class classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(
        in_features=in_features,
        out_features=num_classes,
    )

    return model

def save_model(model, checkpoint_path):
    """Save the model parameters to a checkpoint file."""

    checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

def load_model(model, checkpoint_path, device):
    """Load model parameters from a checkpoint file."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    return model