import json
from pathlib import Path

import matplotlib.pyplot as plt


DATA_PATH = Path("results/confusion_matrix.json")
OUTPUT_PATH = Path("results/confusion_matrix.png")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Confusion matrix data was not found: {DATA_PATH}"
        )

    data = json.loads(
        DATA_PATH.read_text(encoding="utf-8")
    )

    class_names = data["class_names"]
    matrix = data["matrix"]

    figure, axis = plt.subplots(figsize=(7, 6))

    image = axis.imshow(matrix)

    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")

    axis.set_xticks(
        range(len(class_names)),
        labels=class_names,
        rotation=45,
        ha="right",
    )

    axis.set_yticks(
        range(len(class_names)),
        labels=class_names,
    )

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Confusion matrix saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()