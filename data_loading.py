from more_itertools import unzip
from torch.utils.data import DataLoader, Dataset, random_split
import kaggle
import zipfile
import pandas as pd
import os
from PIL import Image
from yaml import load
import torchvision.transforms as tv_transforms


class HistopathologicCancerDetectionDataset(Dataset):
    KAGGLE_DATASET = "histopathologic-cancer-detection"

    RELEVANT_FILES = ["train/", "train_labels.csv"]

    def __init__(self, data_path, download=False, download_path="/tmp", transforms=[]):
        self.data_path = data_path
        self.download_path = download_path
        if download:
            self._download()
        self.train_labels = pd.read_csv(
            os.path.join(self.data_path, "train_labels.csv")
        )
        self.transforms = tv_transforms.Compose(transforms + [tv_transforms.ToTensor()])

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, index):
        image_id = self.train_labels["id"].iloc[index]
        label = self.train_labels["label"].iloc[index]
        img = Image.open(os.path.join(self.data_path, "train", image_id + ".tif"))
        return self.transforms(img), label

    def _download(self):
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(
            self.KAGGLE_DATASET, path=self.download_path, quiet=False
        )
        # TODO: Only get train and labels files, more transfors
        with zipfile.ZipFile(
            os.path.join(self.download_path, self.KAGGLE_DATASET + ".zip")
        ) as zipped:
            for member in filter(lambda name: name.startswith(self.RELEVANT_FILES), zipped.namelist()):
                zipped.extract(member, self.data_path)


def load_data(data_path=None, download=False, test_split=0.33, batch_size=32):
    """
    Downloads the dataset from Kaggle if needed, creates a Pytorch Dataset and then
    setups data loaders for the training and test set.

    Required setup:
    1. Go to https://www.kaggle.com/<ACCOUNT_NAME>/account
    2. Click "Create New API Token"
    3. Add Token file to ~/.kaggle/kaggle.json
    """
    if not data_path:
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    dataset = HistopathologicCancerDetectionDataset(data_path, download=download)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader
