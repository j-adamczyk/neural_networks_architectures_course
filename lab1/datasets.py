"""
Datasets for lab 1.

They are defined in this file, not in the notebook, so that DataLoader worker
processes can always import them. On Windows, workers are started as new
processes, which can't access classes and functions defined in a notebook.
"""

import os
from collections.abc import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.io import ImageReadMode, decode_image

CUB_URL = (
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
)
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-{}.tgz"
IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]  # fmt: skip


class CUB200(Dataset):
    """Caltech-UCSD Birds-200-2011: 200 bird species, official train/test split."""

    def __init__(
        self, root: str, train: bool = True, transform: Callable | None = None
    ) -> None:
        self.base = os.path.join(root, "CUB_200_2011")
        if not os.path.isdir(os.path.join(self.base, "images")):
            download_and_extract_archive(
                CUB_URL, root, filename="CUB_200_2011.tgz", remove_finished=True
            )

        paths = dict(self._read("images.txt"))
        labels = {i: int(c) - 1 for i, c in self._read("image_class_labels.txt")}
        is_train = {i: t == "1" for i, t in self._read("train_test_split.txt")}
        ids = [i for i, _ in self._read("images.txt") if is_train[i] == train]
        self.samples = [
            (os.path.join(self.base, "images", paths[i]), labels[i]) for i in ids
        ]
        self.targets = [label for _, label in self.samples]
        # "001.Black_footed_Albatross" -> "Black footed Albatross"
        self.classes = [
            c.split(".", 1)[1].replace("_", " ") for _, c in self._read("classes.txt")
        ]
        self.transform = transform

    def _read(self, name: str) -> list[list[str]]:
        # lines of a metadata file, split into columns
        with open(os.path.join(self.base, name)) as f:
            return [line.split() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple:
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return (self.transform(img) if self.transform else img), label


def read_rgb_image(path: str) -> torch.Tensor:
    # decode an image file to a uint8 tensor (3, H, W)
    return decode_image(path, mode=ImageReadMode.RGB)


def load_imagenette(
    root: str, split: str, img_size: int, transform: Callable, num_workers: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode all images of an Imagenette split once, in parallel.

    Returns uint8 images (N, 3, H, W) and labels (N,). For images larger than
    160 px, the dataset version with the shorter image side of 320 px is used,
    otherwise the version with 160 px.
    """
    version = "320" if img_size > 160 else "160"
    path = os.path.join(root, f"imagenette2-{version}")
    if not os.path.isdir(path):
        url = IMAGENETTE_URL.format(version)
        download_and_extract_archive(url, root, remove_finished=True)
    folder = ImageFolder(
        os.path.join(path, split), loader=read_rgb_image, transform=transform
    )
    loader = DataLoader(folder, batch_size=256, num_workers=num_workers)
    images, labels = zip(*loader)
    return torch.cat(images), torch.cat(labels)


class TensorImageDataset(Dataset):
    """Images kept in memory as a uint8 tensor, augmented on the fly by transform."""

    def __init__(
        self, images: torch.Tensor, labels: torch.Tensor, transform: Callable
    ) -> None:
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.images[i]), self.labels[i]
