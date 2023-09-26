# %%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl


#TODO: Whole section may be reworked and include more of pytorch lightning tools
# %%
class ImagesDataset(Dataset):
    """
    Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(
        self, x_df: pd.DataFrame, transform: transforms, y_df: pd.DataFrame = None
    ):
        """
        :param pd.DataFrame x_df: links of the jpg
        :param transforms transform: for basic transformation (like normalisation)
        :param pd.DataFrame y_df: labels
        """
        self.data = x_df
        self.label = y_df
        self.transform = transform


    def __getitem__(self, index: int) -> dict:
        """
        :param int index: index of the data path

        :return: dictionary of id,image,label
        :rtype: dict

        """
        # get image from path
        path = r"./data/" + self.data.iloc[index]["filepath"]
        image = Image.open(path).convert("RGB")

        # transform the picture
        image = self.transform(image)
        # get image id from index
        image_id = self.data.index[index]

        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(
                self.label.iloc[index].values, dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        basic_transform: transforms,
        train_features_path: str = "./data/train.csv",
        train_labels_path: str = "./data/label.csv",
        test_features_path: str = "./data/test.csv",
    ) -> None:
        """
        Jan
        :param transforms basic_transform: basic tranformation -> default resize(224,224), ToTensor, standardize
        :param str train_features_path:
        :param str train_labels_path:
        :param str test_features_path:
        """
        # load_data
        self.train_features = pd.read_csv(train_features_path, index_col="id")
        self.train_labels = pd.read_csv(train_labels_path, index_col="id")
        test_features = pd.read_csv(test_features_path, index_col="id")

        # prepare transforms
        self.basic_transform = basic_transform

        # exclude the 2nd transformation in val und test set -> data augmentation only used by trainset
        self.exclude_augmentation_transformer = transforms.Compose(
            basic_transform.transforms[:1] + basic_transform.transforms[1 + 1:])

        # exclude data augmentation compose
        self.test = ImagesDataset(
            test_features, self.exclude_augmentation_transformer)

    def prepare_data(self, fold_number) -> None:
        #TODO: add comments
        val_features = self.train_features.loc[self.train_features["split"]
                                               == fold_number, self.train_features.columns != "split"]
        train_features = self.train_features.loc[self.train_features["split"]
                                                 != fold_number, self.train_features.columns != "split"]

        val_labels = self.train_labels.loc[self.train_labels["split"]
                                           == fold_number, self.train_labels.columns != "split"]
        train_labels = self.train_labels.loc[self.train_labels["split"]
                                             != fold_number, self.train_labels.columns != "split"]

        self.train = ImagesDataset(
            train_features, self.basic_transform, train_labels)
        self.val = ImagesDataset(
            val_features, self.exclude_augmentation_transformer, val_labels)

    def train_dataloader(self, batch_size: int = 128, num_workers: int = 16):
        """
        :param int batch_size: batch size of the training data -> default 64
        :param int num_workers: number of workers for the data loader (optimize if GPU usage not optimal) -> default 16
        """
        return DataLoader(self.train, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    def val_dataloader(self):
        """
        """
        return DataLoader(self.val, batch_size=256)

    def test_dataloader(self):
        """
        """
        return DataLoader(self.test, batch_size=256)

# %%
