# %%
from torchvision import transforms
from torch import nn
import torch
import numpy as np
# %%


class None_Transform(nn.Module):
    """
    Is used as a None Transform, only forwards the inputs
    Was necessary to create the whole composition in a nice way when using Transformer 
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


# %%
class Transformer:
    """ Class which executes a composition of Transformations"""

    def __init__(
        self,
        data_augmentation_transformer: transforms.Compose,
        preprocessing: str,
    ) -> None:
        """
        :param torchvision.transforms data_augmentation_transformer: transformation steps for data_augmentation
        :param str preprocessing: string which defines a preproccessing function
        """
        self.data_augmentation_transformer = data_augmentation_transformer

        # determine preprocessing steps from presets
        if preprocessing == "standard":
            self.preprocessing =  self.standard
        elif preprocessing == "select_roi":
            self.preprocessing =  self.select_roi

    def standard(self,mri):
        return torch.tensor(mri)
    
    def select_roi(self,mri, new_size=(384, 384), crop_size= (112, 112, 6)):
        resize_transform = transforms.Resize(new_size)
        # Process each slice
        resized_slices = []
        for slice_idx in range(mri.shape[2]):
            # Extract the slice and add a channel dimension
            slice = mri[:, :, slice_idx]
            slice = torch.tensor(slice).unsqueeze(0)  # Add a channel dimension
            resized_slice = resize_transform(slice)
            resized_slices.append(resized_slice.squeeze(0).numpy())

        resized_image = np.stack(resized_slices, axis=2)
        center = np.array(resized_image.shape) // 2
        cropped_image = resized_image[
            center[0]-crop_size[0]//2 : center[0]+crop_size[0]//2,
            center[1]-crop_size[1]//2 : center[1]+crop_size[1]//2,
            center[2]-crop_size[2]//2 : center[2]+crop_size[2]//2
        ]
        cropped_image = torch.tensor(cropped_image,dtype=float)

        pixels = cropped_image[cropped_image > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (cropped_image - mean)/std
        return out
    
