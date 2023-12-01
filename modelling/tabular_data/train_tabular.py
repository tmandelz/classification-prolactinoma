from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import wandb
import numpy as np
import seaborn as sns
import numpy as np
import pandas as pd


def setup_wandb_run(
    project_name: str,
    run_group: str,
    fold: int,
    model_architecture: str,
    batchsize: int,
    wandb_additional_config: dict
):
    """
    Sets a new run up (used for k-fold)
    :param str project_name: Name of the project in wandb.
    :param str run_group: Name of the project in wandb.
    :param str fold: number of the executing fold
    :param str model_architecture: Modeltype (architectur) of the model
    :param int batchsize
    :param dict wandb_additional_config: dictionary with additional wandb run parameters
    """
    # init wandb
    run = wandb.init(
        settings=wandb.Settings(start_method="thread"),
        project=project_name,
        entity="pro5d-classification-prolactinoma",
        name=f"{fold}-Fold",
        group=run_group,
        config=wandb_additional_config | {
            "model architecture": model_architecture,
            "batchsize": batchsize,
        }
    )
    return run


def fit(model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.DataFrame,
        X_test: pd.DataFrame, y_test: pd.DataFrame, fold: int, run_group: str = "Tab-Data",
        model_architecture: str = 'LogReg', verbose: bool = False,
        class_names: list = ['non-prolaktinom', 'prolaktinom'],
        wandb_additional_config: dict = {}, learning_curve: bool = True, learning_curve_increase: int = 0):
    """
    Fits a model on data from a fold and evaluates on train and test set
    :param BaseEstimator model: Sklearn BaseEstimator or Implementation
    :param pd.DataFrame X_train: Training Features, should include a 'fold' column
    :param pd.DataFrame y_train: Training Labels
    :param pd.DataFrame X_test: Test Features, should not include a 'fold' column
    :param pd.DataFrame y_test: Test Labels, should include a 'fold' column
    :param int fold: Fold integer to fit on
    :param str run_group: name of the rungroup in wandb
    :param str model_architecture: model architecture of the rungroup in wandb
    :param bool verbose: boolean if metrics should be printed
    :param list class_names: label names as strings, maybe overwritten for some models
    :param dict wandb_additional_config: dictionary with additional wandb run parameters
    """
    # init the wandb run
    run = setup_wandb_run(project_name="tabular-data",
                          run_group=run_group,
                          fold=fold, model_architecture=model_architecture,
                          wandb_additional_config=wandb_additional_config,
                          batchsize='Full',)

    if fold == 'all':
        if verbose:
            print("Using the whole dataset to train.")
        X_train_all_folds = X_train
        y_train_all_folds = y_train
        data_fold = pd.DataFrame()
    else:
        # get the data from the fold and remove the fold column
        data_fold = X_train[X_train['fold'] == fold]
        y_fold = y_train[X_train['fold'] == fold]

        X_train_all_folds = X_train[X_train['fold'] != fold]
        y_train_all_folds = y_train[X_train['fold'] != fold]

        data_fold = data_fold.drop('fold', axis=1)
    # drop fold column
    X_train_all_folds = X_train_all_folds.drop('fold', axis=1)

    if verbose:
        print(f"Shape of validation fold: {data_fold.shape}")
        print(f"Shape of Trainset: {X_train_all_folds.shape}")

    if learning_curve:
        run.log({'train_set_increase': learning_curve_increase,
                'train_set_size': len(X_train_all_folds)})

    # fit the model on train set
    model.fit(X_train_all_folds, y_train_all_folds)
    # evaluate on train set (all folds)
    evaluate_model(model, X_train_all_folds, y_train_all_folds,
                   run, class_names, 'train', verbose)
    if fold != "all":
        # evaluate on val set (left out fold)
        evaluate_model(model, data_fold, y_fold, run,
                       class_names, 'val', verbose)
    # evaluate on test set
    evaluate_model(model, X_test, y_test, run, class_names, 'test', verbose)
    # finish the wandb run
    run.finish()
    return model


def evaluate_model(model: BaseEstimator, X: pd.DataFrame, y_true: pd.DataFrame,
                   wandbrun: wandb.run, class_names: list, eval_mode: str = 'train',
                   verbose: bool = False, class_names_cm: list = ['non-prolaktinom', 'prolaktinom']):
    """
    Fits a model on data from a fold and evaluates on train and test set
    :param BaseEstimator model: Sklearn BaseEstimator or Implementation
    :param pd.DataFrame X: Features to evaluate on
    :param pd.DataFrame y_true: Labels to evaluate on
    :param wandb.run wandbrun: wandb run to log to
    :param str eval_mode: str to indicate that the evaluation is on the training, val set or test set, logs less or more to wandb
    :param bool verbose: boolean if metrics should be printed
    :param list class_names: label names as strings for the metrics, may be changed
    :param list class_names_cm: label names as strings for the reports, should not be changed
    """

    # predict labels als well as probabilities
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)[:, 1]

    # calculate various metrics and reports
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=class_names[-1])
    recall = recall_score(y_true, y_pred, pos_label=class_names[-1])
    f1score = f1_score(y_true, y_pred, pos_label=class_names[-1])
    report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Extract values from the confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()

    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # encode labels for metrics and reports
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_true)
    # calculate fpr and tpr as well as auc
    fpr, tpr, thresholds = roc_curve(y_encoded, y_pred_prob)
    auc = roc_auc_score(y_encoded, y_pred_prob)

    # print the metrics and reports if verbose
    if verbose:
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("f1score:", f1score)
        print(f"Sensitivity: {sensitivity:.2f}")
        print(f"Specificity: {specificity:.2f}")
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", conf_matrix)

    if eval_mode == 'train':
        # Log the train metrics to wandb
        wandbrun.log({'accuracy-train': accuracy, 'precision-train': precision,
                      'recall-train': recall, 'f1score-train': f1score,
                      'sensitivity-train': sensitivity, 'specificity-train': specificity,
                      'auc-train': auc})
    elif eval_mode == 'val':
        # Log the train metrics to wandb
        wandbrun.log({'accuracy-val': accuracy, 'precision-val': precision, 'recall-val': recall,
                      'f1score-val': f1score, 'sensitivity-val': sensitivity, 'specificity-val': specificity,
                      'auc-val': auc})
    elif eval_mode == 'test':
        # Log the test metrics to wandb
        wandbrun.log({'accuracy-test': accuracy, 'precision-test': precision, 'recall-test': recall, 'f1score-test': f1score,
                      'sensitivity-test': sensitivity, 'specificity-test': specificity, 'auc-test': auc})

    if eval_mode != 'train':
        # Log the confusion matrix image to wandb
        plt.figure(figsize=(12, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    cbar=False, xticklabels=class_names_cm,
                    yticklabels=class_names_cm)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        wandbrun.log({f"confusion_matrix-{eval_mode}": wandb.Image(plt)})
        plt.close()

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
        wandbrun.log({f"roc_auc_score--{eval_mode}": wandb.Image(plt)})
        plt.close()

        # Log random sampled qualitative results to wandb
        # sample 5 entries
        random_ind = np.random.choice(range(0, len(X)), 5, replace=False)
        log_data_rows = X.iloc[random_ind]

        log_Y_true = pd.DataFrame(y_true).iloc[random_ind]
        log_Y_pred = pd.DataFrame(pd.DataFrame(y_pred).iloc[random_ind])

        log_Y_true.columns = ['label']
        log_Y_pred.columns = ['prediction']

        correct_pred = pd.DataFrame((log_Y_true.values == log_Y_pred.values))
        correct_pred.columns = ['correct_prediction']
        correct_pred.index = log_Y_true.index
        log_Y_pred.index = log_Y_true.index

        # concat features, label and prediction
        log_df = pd.concat(
            [log_data_rows, log_Y_true, log_Y_pred, correct_pred], axis=1)
        table = wandb.Table(dataframe=log_df)
        wandbrun.log({f"Qualitative-Results-{eval_mode}": table})

        if hasattr(model, 'feature_importances_'):
            # Get feature importance
            data = pd.DataFrame({'Feature': model.feature_names_in_,
                                 'Importance': model.feature_importances_})
            # Sort the DataFrame by the 'Importance' column in descending order
            data = data.sort_values(by='Importance', ascending=False)
            # Create a bar plot
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Importance', y='Feature', data=data,)
            # Add labels and title
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance')
            wandbrun.log({f"feature-importance-{eval_mode}": wandb.Image(plt)})
            plt.close()
