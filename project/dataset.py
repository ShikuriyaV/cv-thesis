import random

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_CLASSES = [
    "Abyssinian",
    "Persian",
    "Beagle",
    "Pug",
]


def build_transforms():
    """Create image transforms for training, validation, and testing."""

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    return train_transform, eval_transform

class PetSubset(Dataset):
    """Use selected samples with a custom transform and remapped labels."""

    def __init__(self, base_dataset, indices, transform, label_map):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        image, original_label = self.base_dataset[original_index]

        if self.transform is not None:
            image = self.transform(image)

        new_label = self.label_map[original_label]

        return image, new_label

def split_indices_by_class(
    labels,
    selected_labels,
    val_ratio=0.2,
    seed=42,
):
    """Split every selected class into training and validation samples."""

    random_generator = random.Random(seed)

    train_indices = []
    val_indices = []

    for label in selected_labels:
        class_indices = [
            index
            for index, sample_label in enumerate(labels)
            if sample_label == label
        ]

        random_generator.shuffle(class_indices)

        val_size = max(1, int(len(class_indices) * val_ratio))

        val_indices.extend(class_indices[:val_size])
        train_indices.extend(class_indices[val_size:])

    random_generator.shuffle(train_indices)
    random_generator.shuffle(val_indices)

    return train_indices, val_indices

def build_datasets(
    data_dir="project/data",
    val_ratio=0.2,
    seed=42,
):
    """Download, filter, and split the Oxford-IIIT Pet dataset."""

    train_transform, eval_transform = build_transforms()

    trainval_base = OxfordIIITPet(
        root=data_dir,
        split="trainval",
        target_types="category",
        download=True,
    )

    test_base = OxfordIIITPet(
        root=data_dir,
        split="test",
        target_types="category",
        download=True,
    )

    original_class_indices = [
        trainval_base.class_to_idx[class_name]
        for class_name in TARGET_CLASSES
    ]

    label_map = {
        original_label: new_label
        for new_label, original_label in enumerate(original_class_indices)
    }

    train_indices, val_indices = split_indices_by_class(
        labels=trainval_base._labels,
        selected_labels=original_class_indices,
        val_ratio=val_ratio,
        seed=seed,
    )

    test_indices = [
        index
        for index, label in enumerate(test_base._labels)
        if label in label_map
    ]

    train_dataset = PetSubset(
        base_dataset=trainval_base,
        indices=train_indices,
        transform=train_transform,
        label_map=label_map,
    )

    val_dataset = PetSubset(
        base_dataset=trainval_base,
        indices=val_indices,
        transform=eval_transform,
        label_map=label_map,
    )

    test_dataset = PetSubset(
        base_dataset=test_base,
        indices=test_indices,
        transform=eval_transform,
        label_map=label_map,
    )

    return train_dataset, val_dataset, test_dataset    

def build_dataloaders(
    data_dir="project/data",
    batch_size=32,
    num_workers=0,
):
    """Create training, validation, and test DataLoaders."""

    train_dataset, val_dataset, test_dataset = build_datasets(
        data_dir=data_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader