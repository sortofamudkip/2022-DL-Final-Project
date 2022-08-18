from more_itertools import unzip
from torch.utils.data import DataLoader, Dataset, random_split
import kaggle
import zipfile
import pandas as pd
import os
from PIL import Image
from yaml import load
import torchvision.transforms as tv_transforms
from utils import KAGGLE_DATASET


def download_dataset(data_path):
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(KAGGLE_DATASET, path=data_path, quiet=False)


class HistopathologicCancerDetectionDataset(Dataset):
    def __init__(
        self, data_path, download=False, transforms=[], first_n_rows=0,
    ):
        if download:
            download_dataset(data_path)
        self.zip_file = zipfile.ZipFile(
            os.path.join(data_path, KAGGLE_DATASET + ".zip")
        )
        self.train_labels = pd.read_csv(self.zip_file.open("train_labels.csv"))

        if (
            first_n_rows and first_n_rows > 0
        ):  # obtain only first N rows of dataset. Used for debugging.
            self.train_labels = self.train_labels.head(first_n_rows)
        self.transforms = tv_transforms.Compose(transforms)

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):
        image_id = self.train_labels["id"].iloc[index]
        label = self.train_labels["label"].iloc[index]
        image_file = self.zip_file.open(os.path.join("train", image_id + ".tif"))
        img = Image.open(image_file)
        return self.transforms(img), label


class HistopathologicCancerDetectionSubmissionDataset(Dataset):
    def __init__(self, data_path, download=False, figsize=224):
        if download:
            download_dataset(data_path)
        self.data_path = data_path
        self.zip_file = zipfile.ZipFile(
            os.path.join(data_path, KAGGLE_DATASET + ".zip")
        )
        self.ids = pd.read_csv(self.zip_file.open("sample_submission.csv")).iloc[:, 0]
        self.transforms = tv_transforms.Compose(
            [tv_transforms.Resize(figsize), tv_transforms.ToTensor()],
        )

    def __len__(self):
        return len(self.test_id)

    def __getitem__(self, index):
        image_id = self.ids.iloc[index]
        image_file = self.zip_file.open(os.path.join("test", image_id + ".tif"))
        img = Image.open(image_file)
        return self.transforms(img), image_id


def load_data(
    data_path=None,
    download=False,
    transforms=[],
    test_split=0.33,
    batch_size=64,
    first_n_rows=0,
):
    """
    Downloads the dataset from Kaggle if needed, creates a Pytorch Dataset and then
    setups data loaders for the training and test set. You can specify torchvision
    transforms to prepare images for the model.

    Required setup:
    1. Go to https://www.kaggle.com/<ACCOUNT_NAME>/account
    2. Click "Create New API Token"
    3. Add Token file to ~/.kaggle/kaggle.json
    """
    if not data_path:
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    dataset = HistopathologicCancerDetectionDataset(
        data_path, download=download, transforms=transforms, first_n_rows=first_n_rows
    )
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_set, shuffle=True, batch_size=batch_size, num_workers=2
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=2)
    return train_loader, test_loader


def load_submission_data(
    data_path=None, download=False, transforms=[], batch_size=64,
):
    """
    Creates data_loader for test dataset
    """
    if not data_path:
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test")
    dataset = HistopathologicCancerDetectionSubmissionDataset(
        data_path, download=download, transforms=transforms
    )

    submission_loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    return submission_loader
