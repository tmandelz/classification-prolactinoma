from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score, classification_report, confusion_matrix,roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import wandb
import seaborn as sns

def setup_wandb_run(
    project_name: str,
    run_group: str,
    fold: int,
    model_architecture: str,
    batchsize:int,
):
    """
    Sets a new run up (used for k-fold)
    :param str project_name: Name of the project in wandb.
    :param str run_group: Name of the project in wandb.
    :param str fold: number of the executing fold
    :param str model_architecture: Modeltype (architectur) of the model
    :param int batchsize
    :param int seed
    """
    # init wandb
    run = wandb.init(
        settings=wandb.Settings(start_method="thread"),
        project=project_name,
        entity="pro5d-classification-prolactinoma",
        name=f"{fold}-Fold",
        group=run_group,
        config={
            "model architecture": model_architecture,
            "batchsize":batchsize,
        },
    )
    return run

def fit(model:BaseEstimator,X_train,y_train,X_test,y_test, fold:int,run_group="Tab-Data",
        model_architecture='LogReg',preprocessing_func = None,
        verbose:bool = False,class_names = ['non-prolaktinom','prolaktinom']):
    run = setup_wandb_run(project_name="pro5d-classification-prolactinoma",
                        run_group=run_group,
                        fold=fold, model_architecture=model_architecture,
                          batchsize='Full')
    if preprocessing_func is not None:
        X_train = preprocessing_func(X_train)
    
    data_fold = X_train[X_train['fold'] == fold]
    y_fold = y_train[X_train['fold'] == fold]
    data_fold = data_fold.drop('fold', axis=1)
    
    model.fit(data_fold,y_fold)
    evaluate_model(model,data_fold,y_fold,run,class_names,True,verbose)
    evaluate_model(model,X_test,y_test,run,class_names,False,verbose)
    run.finish()
    return model

def evaluate_model(model:BaseEstimator,data,y_true,wandbrun,class_names,train:bool =False,verbose:bool = False,class_names_cm = ['non-prolaktinom','prolaktinom']):

    y_pred = model.predict(data)
    y_pred_prob = model.predict_proba(data)[:, 1]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred,pos_label=class_names[-1])
    recall = recall_score(y_true, y_pred,pos_label=class_names[-1])
    f1score = f1_score(y_true, y_pred,pos_label=class_names[-1])
    report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Extract values from the confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()

    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    
    

    if verbose:
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("f1score:", f1score)
        print(f"Sensitivity: {sensitivity:.2f}")
        print(f"Specificity: {specificity:.2f}")
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", conf_matrix)
    
    if train:
        wandbrun.log({'accuracy-train':accuracy})
        wandbrun.log({'precision-train':precision})
        wandbrun.log({'recall-train':recall})
        wandbrun.log({'f1score-train':f1score})
        wandbrun.log({'sensitivity-train':sensitivity})
        wandbrun.log({'specificity-train':specificity})
    else:
        wandbrun.log({'accuracy-test':accuracy})
        wandbrun.log({'precision-test':precision})
        wandbrun.log({'recall-test':recall})
        wandbrun.log({'f1score-test':f1score})
        wandbrun.log({'sensitivity-test':sensitivity})
        wandbrun.log({'specificity-test':specificity})

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_true) 
        
        fpr, tpr, thresholds = roc_curve(y_encoded, y_pred_prob)
        auc = roc_auc_score(y_encoded, y_pred_prob) 
        wandbrun.log({'auc-test':auc})

        plt.figure(figsize=(12, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names_cm, yticklabels=class_names_cm)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # Log the confusion matrix image to W&B
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.axhline(y=0.8, color='r', linestyle='--', label='sensitivity at 80%', alpha=0.3)
        plt.axvline(x=0.7, color='g', linestyle='--', label='specificity at 70%', alpha=0.3)
        plt.legend(loc="lower right")
        wandb.log({"roc_auc_score": wandb.Image(plt)})
        plt.close()
