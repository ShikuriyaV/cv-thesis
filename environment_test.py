"""Verify the Python, PyTorch, and CUDA environment."""

import sys

import torch
import torchvision


def main() -> None:
    """Print environment information and perform a tensor calculation."""

    print("=" * 55)
    print("Environment verification")
    print("=" * 55)

    print("Python version:", sys.version.split()[0])
    print("Python executable:", sys.executable)
    print("PyTorch version:", torch.__version__)
    print("TorchVision version:", torchvision.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Selected device:", device)
        print("GPU name:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "GB",
        )
    else:
        device = torch.device("cpu")
        print("Selected device:", device)

    # Create tensors directly on the selected device.
    x = torch.rand(2, 3, device=device)
    y = torch.rand(3, 2, device=device)

    result = x @ y

    print("\nTensor x:")
    print(x)

    print("\nTensor y:")
    print(y)

    print("\nMatrix multiplication result:")
    print(result)

    print("\nResult device:", result.device)
    print("Environment test passed successfully.")


if __name__ == "__main__":
    main()