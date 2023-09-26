# %%
from torchvision import transforms
from torch import nn
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
        preprocessing_transformer: str,
        pretrained_transformer: str,
    ) -> None:
        """
        Initialises a Transformer class for our project.
        this class uses presets for a transfer learning approaches and presets for preprocessing approaches.
        it also uses a parameterized data augmentation set 
        :param torchvision.transforms data_augmentation_transformer: transformation steps for data_augmentation
        :param str preprocessing_transformer: string which defines a preset for all our transformation steps (resizing etc.)
        :param str pretrained_transformer: string which defines a preset for the pretrained transfer model settings
        """
        self.pretrained_transformer = pretrained_transformer
        self.preprocessing_transformer = preprocessing_transformer
        self.data_augmentation_transformer = data_augmentation_transformer

        # determine preprocessing steps from presets
        if preprocessing_transformer == "standard":
            self.preprocessing_transformer = transforms.Compose(
                [
                    transforms.Resize((224, 224), antialias=True),
                ]
            )
        elif preprocessing_transformer == "overfitting":
            self.preprocessing_transformer = transforms.Compose(
                [
                    transforms.Resize((20, 20), antialias=True),
                ]
            )
        elif preprocessing_transformer == "model_specific":
            self.preprocessing_transformer = transforms.Compose(
                [
                    None_Transform()
                ]
            )

        # determine transformation steps for pretrained Models presets
        if pretrained_transformer == "efficientnet":
            self.pretrained_transformer = transforms.Compose(
                [
                    transforms.Resize(
                        (256, 256), antialias=True, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif pretrained_transformer == "resnet":
            self.pretrained_transformer = transforms.Compose(
                [
                    transforms.Resize(
                        (232, 232), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif pretrained_transformer == "mlp":
            self.pretrained_transformer = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            )
        elif pretrained_transformer == "convnext":
            self.pretrained_transformer = transforms.Compose(
                [
                    transforms.Resize(
                        (236, 236), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
        elif pretrained_transformer == "swin":
            self.pretrained_transformer = transforms.Compose(
                [
                    transforms.Resize(
                        (232, 232), antialias=True, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
        elif pretrained_transformer == "inceptionv3":
            self.pretrained_transformer = transforms.Compose(
                [
                    transforms.Resize(
                        (342, 342), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(299),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])

    def getCompose(self, turn_off_to_tensor: bool = False) -> transforms.Compose:
        """
        Returns a Composition of Preprocessing Transformations for different transfer learning and augmentation approaches
        :param bool turn_off_to_tensor: don't use transforms.ToTensor() for certain augmentations
        :return: transforms.Compose object representing a sequence of image transformations for the full preprocessing
        """
        # some augmentations, for example ColorJitter, do not accept a tensor, others do
        # we eather first turn everything into a tensor or do it after the augmentation transformation
        if turn_off_to_tensor:
            first_trans = None_Transform()
            sec_trans = transforms.ToTensor()
        else:
            first_trans = transforms.ToTensor()
            sec_trans = None_Transform()
        return transforms.Compose(
            [
                # add a to Tensor in front off all augmentations
                first_trans,
                # first execute the augmenation steps
                self.data_augmentation_transformer,
                # for special data augmentation methods
                sec_trans,
                # second execute the preprocessing steps
                self.preprocessing_transformer,
                # third execute the pretrained model step
                self.pretrained_transformer,
            ]
        )
