import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from PIL import Image
from strenum import StrEnum


class Label(StrEnum):
    """
    Enumeration String for our labels
    """
    Label_1 = "label_1"
    Label_2 = "label_2"
    Label_x = "label_x"



class Evaluation:
    def __init__(self, data_classes: list = [label.value for label in Label]) -> None:
        """
        Jan
        :param list data_classes: list of true labels (convert from int to str)
        """
        self.classes = data_classes

    def per_batch(self, index_batch: int, epoch: int, loss_batch: float, loss_val: float = None) -> None:
        """
        Thomas
        Logs the loss of a batch
        :param int index_batch: index of the batch to log (step)
        :param int epoch: index of the epoch to log
        :param float loss_batch: loss of the batch for the trainset
        :param float loss_val: loss of the batch for the validationset, is not logged every epoch, defaults to None
        """
        wandb.log({"index_batch": index_batch,
                  "epoch": epoch, "loss batch": loss_batch, "loss batch val": loss_val})

    def per_epoch(
        self,
        epoch: int,
        loss_train: float,
        pred_train: np.array,
        label_train: np.array,
        loss_val: float,
        pred_val: np.array,
        label_val: np.array,
    ) -> None:
        """
        Jan
        wandb log of different scores
        :param int epoch: index of the epoch to log
        :param float loss_train: log loss of the training
        :param np.array pred_train: prediction of the training
        :param np.array label_train: labels of the training
        :param float loss_val: log loss of the validation
        :param np.array pred_val: prediction of the validation
        :param np.array label_val: labels of the validation
        """
        f1_train = {
            f"train f1_score von {self.classes[animal]}": f1_score(
                label_train[:, animal], pred_train == animal
            )
            for animal in range(len(self.classes))
        }
        f1_test = {
            f"validation f1_score von {self.classes[animal]}": f1_score(
                label_val[:, animal], pred_val == animal
            )
            for animal in range(len(self.classes))
        }
        log = {"epoch": epoch, "Loss train": loss_train, "Loss val": loss_val}
        wandb.log({**f1_train, **f1_test, **log})

    def per_model(self, label_val, pred_val, val_data) -> None:
        """
        Jan
        wandb log of a confusion matrix and plots of wrong classified animals
        :param np.array label_val: labels of the validation
        :param np.array pred_val: prediction of the validation
        :param pd.dataframe val_data: validation data
        """
        self.true_label = np.argmax(label_val, axis=1)
        self.true_pred = np.argmax(pred_val, axis=1)
        wrong_classified = np.where(self.true_label != self.true_pred)[0]

        #TODO: add a if, which decides if these should be plotted
        self.plot_16_pictures(
            np.random.choice(wrong_classified, replace=False,
                             size=16), val_data
        )

        wandb.log(
            {
                "confusion matrix": wandb.sklearn.plot_confusion_matrix(
                    self.true_label, self.true_pred, self.classes
                ),
                "wrong prediction": plt,
            }
        )
        plt.close()

        #TODO: add a if, which decides if these should be plotted
        #TODO: generalise "site"
        data_wrong_class = val_data.iloc[wrong_classified]
        site_most_wrong = data_wrong_class[
            data_wrong_class["site"] == data_wrong_class["site"].value_counts(normalize=True
                                                                              ).index[0]
        ]
        if len(site_most_wrong) < 16:
            self.plot_16_pictures(range(len(site_most_wrong)), data_wrong_class)
        else:
            self.plot_16_pictures(
                np.random.choice(range(len(site_most_wrong)),
                                 size=16, replace=False),
                data_wrong_class,
            )

        plt.suptitle("worst site: " +
                     str(data_wrong_class["site"][0]), size=120)
        wandb.log({"Bad site": plt})
        plt.close()

    def plot_16_pictures(self, index: np.array, data: pd.DataFrame) -> None:
        """
        Jan
        plot 16 pcitures
        :param np.array index: index of the chosen observations
        :param pd.DataFrame data: data with the filepath of the images
        """
        #TODO: check and maybe rework this
        fig = plt.figure(figsize=(120, 90), dpi=20)
        for n, variable in enumerate(index):
            ax = fig.add_subplot(4, 4, n + 1)
            datapoint = data.iloc[variable]
            ax.imshow(Image.open(
                "./competition_data/" + datapoint["filepath"]))
            ax.set_title(
                f"{self.classes[self.true_pred[variable]]} anstatt {self.classes[self.true_label[variable]]}",
                size=60,
            )
