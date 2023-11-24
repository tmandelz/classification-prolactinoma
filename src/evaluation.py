import numpy as np
import wandb
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
from ipywidgets import interact
from matplotlib.colors import ListedColormap
import os
import pandas as pd
import numpy as np
from src.augmentation import Transformer,None_Transform
from src.data_modules import DataModule


class Evaluation:

    def per_batch(self, index_batch: int, epoch: int, loss_batch: float) -> None:
        """
        Thomas
        Logs the loss of a batch
        :param int index_batch: index of the batch to log (step)
        :param int epoch: index of the epoch to log
        :param float loss_batch: loss of the batch for the trainset
        """
        print({"index_batch": index_batch,"epoch": epoch, "loss batch": loss_batch})
        #wandb.log({"index_batch": index_batch,"epoch": epoch, "loss batch": loss_batch})

    def per_epoch(
        self,
        epoch: int,
        loss_train: float,
        pred_train: np.array,
        label_train: np.array,
        loss_val: float,
        pred_val: np.array,
        label_val: np.array,
        eval_data: str
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
        :paraam str eval_data: val or test
        """
        true_pred_val = np.round(pred_val)

        sensitivity,specificity,fpr,tpr,auc,conf_matrix = self.calc_metrics(label_val,true_pred_val,pred_val)
        log_val = {f'sensitivity_{eval_data}': sensitivity,
                   f'specificity_{eval_data}': specificity,
                   f'auc_{eval_data}': auc}
        
        true_pred_train = np.round(pred_train)
        sensitivity,specificity,fpr,tpr,auc,conf_matrix = self.calc_metrics(label_train,true_pred_train,pred_train)
        log_train = {'sensitivity_train': sensitivity,
                   'specificity_train': specificity,
                   'auc_train': auc}

        log = {"epoch": epoch, "Loss train": loss_train, f"Loss {eval_data}": loss_val}
        print({**log,**log_val,**log_train})
        # wandb.log({**log,**log_val,**log_train})


    def per_model(self, label_val, pred_val,eval_data:str = "eval") -> None:
        """
        wandb log of a confusion matrix and plots of wrong classified animals
        :param np.array label_val: labels of the validation
        :param np.array pred_val: prediction of the validation
        :param str eval_data: validation data 
        """
        self.eval_data = eval_data
        self.true_pred = np.round(pred_val)
        sensitivity,specificity,fpr,tpr,auc,conf_matrix = self.calc_metrics(label_val,self.true_pred,pred_val)
        print({f'sensitivity {eval_data}': sensitivity,
                f'specificity {eval_data}': specificity,
                f'auc {eval_data}': auc})
        # wandb.log({'sensitivity': sensitivity,
        #            'specificity': specificity,
        #            'auc': auc})
        self.plot_roc_curve(fpr,tpr,auc)
        self.plot_conf_matrix(conf_matrix)

        index_fp = np.where((label_val == 0) & (self.true_pred==1))[0][:5]
        index_fn = np.where((label_val == 1) & (self.true_pred==0))[0][:5]
        index_tp = np.where((label_val == 1) & (self.true_pred==1))[0][:5]
        index_tn = np.where((label_val == 0) & (self.true_pred==0))[0][:5]
        print({f"index false positiv {eval_data}": index_fp,
                   f"index false negativ {eval_data}": index_fn,
                   f"index true positiv {eval_data}":index_tp,
                   f"index true negativ {eval_data}": index_tn})
        # wandb.log({"index false positiv": index_fp,
        #            "index false negativ": index_fn,
        #            "index true positiv":index_tp,
        #            "index true negativ": index_tn})

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
        plt.show()
        #wandb.log({f"roc_auc_score {self.eval_data}": wandb.Image(plt)})
        plt.close()

    def plot_conf_matrix(self,conf_matrix):
        plt.figure(figsize=(12, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    cbar=False, xticklabels=["non prolactinoma","prolactinoma"],
                    yticklabels=["non prolactinoma","prolactinoma"])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
        #wandb.log({f"confusion_matrix {self.eval_data}": wandb.Image(plt)})
        plt.close()
        
    def visualize_slices(self,index:list,fold,mri_type:str ="t1_tse_fs_cor" ,preprocess_slices=None):
        base_transformer = Transformer(
            None_Transform(), "standard"
        )
        data_module = DataModule(base_transformer,mri_type = mri_type)
        data_module.prepare_data(fold)
        if type(fold) == int:
            data = data_module.val
        elif fold == "test":
            data = data_module.test
        else:
            print("Select a int value between 0 - 4 or write 'test'")
            raise ValueError
        
        mri_list =[data.__getitem__(mri_index)["image"] for mri_index in index]
        if preprocess_slices !=None:
            mri_list_processed = list(map(preprocess_slices,mri_list))
            n_slices = mri_list_processed[0].shape[2]
        else:
            mri_list_processed = None
        cmap = plt.cm.winter
        # Get the colormap colors
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
        my_cmap = ListedColormap(my_cmap)
        
        # Function to visualize a single slice
        def show_slice(mri, slice_number):
            if mri_list_processed == None:
                plt.imshow(mri_list[mri][:, :, slice_number], cmap='gray')
                
            else:
                starting_point = mri_list[mri].shape[2]//2 - n_slices//2
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with 2 subplots
                axes[0].imshow(mri_list[mri][:, :, slice_number], cmap='gray')
                axes[0].set_title(f'Not Preprocessed')
                if (starting_point <= slice_number) and (starting_point + n_slices > slice_number):
                    axes[1].imshow(mri_list_processed[mri][:, :, slice_number-starting_point], cmap='gray')
                    axes[1].set_title(f'Preprocessed')
            plt.show()

        
        interact(show_slice, mri=(0, len(mri_list) - 1), slice_number=(0, mri_list[0].shape[2] - 1))