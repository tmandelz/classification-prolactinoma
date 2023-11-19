import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import nibabel as nib

class ImagesDataset(Dataset):
    """
    Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(
        self, data: pd.DataFrame, 
        transformer,
        mri_type: str = "t2_tse_fs_cor",
        use_mri_images: bool=True,
        use_tabular_data: bool= False
    ):
        """
        :param pd.DataFrame x_df: links of the jpg
        :param transformer: save the preproccessing and the augmentation function
        :param str mri_type: type of mri
        :param bool use_mri_images: True if the mri images is used
        :param bool use_tabular_data: True if the tabular data is used
        """
        self.data = data
        self.label = self.data.loc[:,["Patient_ID","Case_ID","Category"]]
        self.preproccessing = transformer.preprocessing
        self.augmentation = transformer.data_augmentation_transformer
        self.mri_type = mri_type
        self.use_mri_images = use_mri_images
        self.use_tabular_data = use_tabular_data


    def __getitem__(self, index: int) -> dict:
        """
        :param int index: index of the data path

        :return: dictionary of id,image,label
        :rtype: dict

        """
        if self.use_mri_images:
            # get mri
            mri_case = self.data.iloc[index]["MRI_Case_ID"]
            mri = self.load_mri(mri_case)

            # transform the picture
            mri = self.preproccessing(mri)
        
            if self.augmentation != None:
                self.augmentation(mri)
        else:
            mri_case = None
        if self.use_tabular_data:
            tab_data = None
        else:
            tab_data = None
        
        case = self.data.iloc[index]["Case_ID"]
        label = self.label.loc[index,"Category"]
        sample = {"case_id": case, "image": mri, "label": label,"tab_data":tab_data}
        return sample

    def __len__(self):
        return len(self.data)
    
    def load_mri(self,mri_case):
        path = f"../raw_data/nii_files/{self.mri_type}/{mri_case}.nii"
        return nib.load(path).get_fdata()

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        augmentation: transforms = lambda x:x, 
        preproccessing:function = None,
        train_data_path: str = "./data/train_data.csv",
        test_data_path: str = "./data/test_data.csv",
        mri_type: str = "t2_tse_fs_cor",
        use_mri_images: bool=True,
        use_tabular_data: bool= False
        
    ) -> None:
        """
        Jan
        :param transforms augmentation: augmentation for train data
        :param function preproccessing: basic preproccesing for ROI-exctraction
        :param str train_data_path:
        :param str test_data_path:
        :param str mri_type: type of mri
        :param bool use_mri_images: True if the mri images is used
        :param bool use_tabular_data: True if the tabular data is used
        """
        
        # load_data
        self.train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)
        
        # prepare transforms
        self.preproccessing = preproccessing,
        self.augmentation = augmentation
        self.mri_type = mri_type
        self.use_mri_images = use_mri_images
        self.use_tabular_data = use_tabular_data

        
        self.test = ImagesDataset(
            test_data, None,self.preproccessing,self.mri_type,self.use_mri_images,self.use_tabular_data)

    def prepare_data(self, fold_number) -> None:

        val_data = self.train_data.loc[train_data["fold"] == fold_number,:]
        train_data = self.train_data.loc[train_data["fold"] != fold_number,:]

        self.train = ImagesDataset(
            train_data,self.augmentation,self.preproccessing,self.mri_type,self.use_mri_images,self.use_tabular_data)
        self.val = ImagesDataset(
            val_data,None,self.preproccessing,self.mri_type,self.use_mri_images,self.use_tabular_data)

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
