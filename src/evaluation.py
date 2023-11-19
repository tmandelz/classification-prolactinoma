import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
from strenum import StrEnum
import seaborn as sns

class Label(StrEnum):
    """
    Enumeration String for our labels
    """
    Label_1 = "label_1"
    Label_2 = "label_2"



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
        wandb log of a confusion matrix and plots of wrong classified animals
        :param np.array label_val: labels of the validation
        :param np.array pred_val: prediction of the validation
        :param pd.dataframe val_data: validation data
        """
        self.true_pred = np.round(pred_val, axis=1)
        sensitivity,specificity,fpr,tpr,auc,conf_matrix = self.calc_metrics(label_val,self.true_pred,pred_val)
        wandb.log({'sensitivity': sensitivity,
                   'specificity': specificity,
                   'auc': auc})
        self.plot_roc_curve(fpr,tpr,auc)
        self.plot_conf_matrix(conf_matrix)

        wrong_classified = np.random.choice(np.where(label_val != self.true_pred)[0])
        correct_classified = np.random.choice(np.where(label_val == self.true_pred)[0])
        

    def calc_metrics(self,y_true,y_pred,y_pred_prob):
        conf_matrix = confusion_matrix(y_true, y_pred)
        # Extract values from the confusion matrix
        tn, fp, fn, tp = conf_matrix.ravel()

        # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        # calculate fpr and tpr as well as auc
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        auc = roc_auc_score(y_true, y_pred_prob)
        return sensitivity,specificity,fpr,tpr,auc,conf_matrix
    
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
        wandb.log({"Wrong Predicted":plt})
        plt.close()

    def plot_roc_curve(self,fpr,tpr,auc):
        # Log AUC Curve to wandb
        plt.figure(figsize=(12, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')

        # Define the coordinates for the shaded region
        x_start, x_end = 0., .3
        y_start, y_end = 0.8, 1

        # Find the overlapping region
        x_overlap = np.clip([x_start, x_end], min(fpr), max(fpr))
        y_overlap = np.clip([y_start, y_end], min(tpr), max(tpr))

        # Add overlapping shading
        plt.fill_between(
            x_overlap, y_overlap[0], y_overlap[1], color='gray',
            alpha=0.3, label='Preferred Area')
        plt.axhline(y=0.8, color='r', linestyle='--',
                    label='sensitivity at 80%', alpha=0.3)
        plt.axvline(x=(1-0.7), color='g', linestyle='--',
                    label='specificity at 70%', alpha=0.3)
        plt.legend(loc="lower right")
        wandb.log({f"roc_auc_score": wandb.Image(plt)})
        plt.close()

    def plot_conf_matrix(self,conf_matrix):
        plt.figure(figsize=(12, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    cbar=False, xticklabels=["non prolactinoma","prolactinoma"],
                    yticklabels=["non prolactinoma","prolactinoma"])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        wandb.log({f"confusion_matrix": wandb.Image(plt)})
        plt.close()