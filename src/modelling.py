# %%
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data_modules import DataModule
from src.evaluation import Evaluation

# %%


def _set_seed(seed: int = 42):
    """
    Jan
    Function to set the seed for the gpu and the cpu
    private method, should not be changed
    :param int seed: DON'T CHANGE
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %%
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class DeepModel_Trainer:
    def __init__(
        self,
        data_model: DataModule,
        model: nn.Module,
        device: torch.device = device,
        random_cv_seeds: list = [42, 43, 44, 45, 46]
    ) -> None:
        """
        Jan
        Load the train/test/val data.
        :param DataModule data_model: instance where the 3 dataloader are available.
        :param nn.Module model: pytorch deep learning module
        :param torch.device device: used device for training
        :param list random_cv_seeds: what random seeds to use
        """
        self.random_cv_seeds = random_cv_seeds
        _set_seed()

        self.data_model = data_model
        self.test_loader = data_model.test_dataloader()
        self.model = model
        self.device = device
        self.evaluation = Evaluation()

    def setup_wandb_run(
        self,
        project_name: str,
        run_group: str,
        fold: int,
        lr: float,
        num_epochs: int,
        model_architecture: str,
        batchsize: int,
    ):
        """
        Sets a new run up (used for k-fold)
        :param str project_name: Name of the project in wandb.
        :param str run_group: Name of the project in wandb.
        :param str fold: number of the executing fold
        :param int lr: learning rate of the model
        :param int num_epochs: number of epochs to train
        :param str model_architecture: Modeltype (architectur) of the model
        :param int batchsize
        """
        # init wandb
        self.run = wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project=project_name,
            entity="pro5d-classification-prolactinoma",
            name=f"{fold}-Fold",
            group=run_group,
            config={
                "learning rate": lr,
                "epochs": num_epochs,
                "model architecture": model_architecture,
                "batchsize": batchsize,
            },
        )

    def train_model(
        self,
        run_group: str,
        model_architecture: str,
        num_epochs: int,
        loss_module: nn = nn.CrossEntropyLoss(),
        test_model: bool = False,
        cross_validation: bool = True,
        project_name: str = "deep_model_x",
        batchsize_train_data: int = 64,
        num_workers: int = 16,
        lr: float = 1e-3,
        validate_batch_loss_each: int = 20,
        cross_validation_random_seeding=False,
        use_mri_images: bool=True,
        use_tabular_data: bool= False
    ) -> None:
        """
        Jan
        To train a pytorch model.
        :param str run_group: Name of the run group (kfolds).
        :param str model_architecture: Modeltype (architectur) of the model
        :param int num_epochs: number of epochs to train
        :param nn.CrossEntropyLoss loss_module: Loss used for the competition
        :param int test_model: If true, it only loops over the first train batch and it sets only one fold. -> For the overfitting test.
        :param int cross_validation: If true, creates 5 cross validation folds to loop over, else only one fold is used for training
        :param str project_name: Name of the project in wandb.
        :param int batchsize: batchsize of the training data
        :param int num_workers: number of workers for the data loader (optimize if GPU usage not optimal) -> default 16
        :param int lr: learning rate of the model
        :param int validate_batch_loss_each: defines when to log validation loss on the batch
        :param bool cross_validation_random_seeding: defines whether to use the same seed for each fold or to use different ones
        :param bool use_mri_images: True if the mri images is used
        :param bool use_tabular_data: True if the tabular data is used
        """
        #TODO: rework and separate cv and evaluation from this whole function to subfunctions
        # train loop over folds
        if cross_validation:
            n_folds = 5
        else:
            n_folds = 1
        self.models = []
        for fold in tqdm(range(n_folds), unit="fold", desc="Fold-Iteration"):
            # set a different random seed for each fold to introduce some variance
            if cross_validation_random_seeding:
                _set_seed(self.random_cv_seeds[fold])

            # setup a new wandb run for the fold -> fold runs are grouped by name
            self.setup_wandb_run(
                project_name,
                run_group,
                fold,
                lr,
                num_epochs,
                model_architecture,
            )

            # prepare the kfold and dataloaders
            self.data_model.prepare_data(fold)
            self.train_loader = self.data_model.train_dataloader(
                batchsize_train_data, num_workers
            )
            self.val_loader = self.data_model.val_dataloader()

            # Overfitting Test for first batch
            if test_model:
                self.train_loader = [next(iter(self.train_loader))]

            # prepare the model
            model = self.model()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # training mode
            model.train()
            model.to(device)
            # train loop over epochs
            batch_iter = 1
            for epoch in tqdm(range(num_epochs), unit="epoch", desc="Epoch-Iteration"):
                loss_train = np.array([])
                label_train_data = np.empty([])
                pred_train_data = np.array([])

                # train loop over batches
                for batch in self.train_loader:

                    # calc gradient
                    data_labels = batch["label"].to(device)
                    if use_mri_images and use_tabular_data:
                        data_inputs = batch["image"].to(device)
                        tab_data = batch["tab_data"].to(device)
                        preds = model(data_inputs,tab_data)
                    elif use_mri_images:
                        data_inputs = batch["image"].to(device)
                        preds = model(data_inputs)
                    elif use_tabular_data:
                        tab_data = batch["tab_data"].to(device)
                        preds = model(tab_data)
                    else:
                        print("No Datatype selected")
                        raise ValueError
                    
                    loss = loss_module(preds, data_labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_val_batch = None
                    if batch_iter % validate_batch_loss_each == 0:
                        pred_val, label_val = self.predict(
                            model,
                            self.val_loader,
                       )
                        loss_val_batch = loss_module(
                            torch.tensor(pred_val), torch.tensor(label_val)
                        )

                    self.evaluation.per_batch(
                        batch_iter, epoch, loss, loss_val_batch)

                    # data for evaluation
                    label_train_data = np.concatenate(
                        (label_train_data, data_labels.data.cpu().numpy()), axis=0
                    )
                    predict_train = torch.argmax(preds, 1).data.cpu().numpy()
                    pred_train_data = np.concatenate(
                        (pred_train_data, predict_train), axis=0
                    )
                    loss_train = np.append(loss_train, loss.item())

                    # iter next batch
                    batch_iter += 1

                # wandb per epoch
                pred_val, label_val = self.predict(
                    model,
                    self.val_loader,
                )
                loss_val = loss_module(torch.tensor(
                    pred_val), torch.tensor(label_val))
                self.evaluation.per_epoch(
                    epoch,
                    loss_train.mean(),
                    pred_train_data,
                    label_train_data,
                    loss_val,
                    pred_val,
                    label_val,
                )

            # wandb per model
            self.evaluation.per_model(
                label_val, pred_val, self.data_model.val.data)

            self.models.append(model)
            self.run.finish()
            self.model_fold5 = model
            self._save_model(str(run_group+str(fold)))
        # new model instance for a new k-fold
        self.model_fold5 = model

    def predict(
        self,
        model: nn.Module,
        data_loader: DataLoader,
    ):
        """
        Jan
        Prediction for a given model and dataset
        :param nn.Module model: pytorch deep learning module
        :param DataLoader data_loader: data for a prediction
        :param int decrease_confidence: devide the output bevor calculating the softmax

        :return: predictions and true labels
        :rtype: np.array, np.array
        """
        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():  # Deactivate gradients for the following code
            for batch in data_loader:
                # Determine prediction of model
                data_inputs = batch["image"].to(self.device)

                preds = model(data_inputs)
                predictions = np.concatenate(
                    (predictions, preds.data.cpu().numpy()), axis=0
                )

                
                data_labels = batch["label"].to(self.device)
                true_labels = np.concatenate(
                    (true_labels, data_labels.data.cpu().numpy()), axis=0
                )
        model.train()
        return predictions, true_labels

    def _save_model(self, submit_name: str):
        """
        Thomas
        saves the models state
        :param str submit_name: name of the model file
        """
        #TODO: rename and rework this function to a save model function. maybe leave submission but use an if statement to make it optional to submit something
        path = f"./model_submit/{submit_name}.pth"
        torch.save(self.model_fold5.state_dict(), path)
        print(f"Saved model: {submit_name} to {path}")
# %%
