from more_itertools import unzip
from torch.utils.data import DataLoader, Dataset, random_split
import kaggle
import zipfile
import pandas as pd
import os
from PIL import Image
from yaml import load
from torchvision import transforms


class HistopathologicCancerDetectionDataset(Dataset):
    KAGGLE_DATASET = "histopathologic-cancer-detection"

    def __init__(self, data_path):
        self.data_path = data_path
        self._download()
        self.train_labels = pd.read_csv(
            os.path.join(self.data_path, "train_labels.csv")
        )
        self.transforms = transforms.Compose([transforms.ToTensor()])

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
            self.KAGGLE_DATASET, path=self.data_path, quiet=False
        )
        # Only unzip if we cannot find certain files
        if not set(["train", "train_labels.csv"]).issubset(os.listdir(self.data_path)):
            with zipfile.ZipFile(
                os.path.join(self.data_path, self.KAGGLE_DATASET + ".zip")
            ) as zipped:
                zipped.extractall(self.data_path)


def load_data(data_path=None, test_split=0.33, batch_size=32):
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
    dataset = HistopathologicCancerDetectionDataset(data_path)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader
