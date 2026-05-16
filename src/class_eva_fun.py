import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler , OneHotEncoder , OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score , confusion_matrix , ConfusionMatrixDisplay ,classification_report ,recall_score ,precision_score

def classification_metrics(y_true , y_pred , target_names ,label='', output_dict = False , fig_size = (8,4) , normalize = 'true' , cmap='Blues' , colorbar = False):
    report = classification_report(y_true , y_pred ,target_names=target_names)
    header = '-'*70
    print(header ,f"Classification metrics : {label}" , header ,sep="\n")
    print(report)
    fig ,axes = plt.subplots(figsize=fig_size , ncols=2)
    ConfusionMatrixDisplay.from_predictions(y_true , y_pred ,cmap='gist_gray',normalize=None , colorbar=colorbar ,ax=axes[0],display_labels=target_names)
    axes[0].set_title("Confusion Matrix")
     
    ConfusionMatrixDisplay.from_predictions(y_true , y_pred ,cmap=cmap ,normalize=normalize , colorbar=colorbar ,ax=axes[1],display_labels=target_names)
    axes[1].set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    plt.show()
    if output_dict == True :
        report_dict = classification_report(y_true , y_pred ,output_dict=output_dict,target_names=target_names)
        return report_dict
    
    
def evaluation_classification(model , X_train , y_train , X_test , y_test , target_names ,figsize = (6,4) , normalize= 'true' , output_dict = False , cmap_train = 'Blues' , cmap_test = 'Reds' , colorbar = False):
    y_train_pred = model.predict(X_train)
    result_train = classification_metrics(y_train , y_train_pred ,output_dict=output_dict , fig_size=figsize , cmap=cmap_train , normalize=normalize , colorbar=colorbar ,label='Training Data' ,target_names=target_names)
    print()
    y_test_pred = model.predict(X_test)
    result_test = classification_metrics(y_test ,y_test_pred ,output_dict=output_dict , fig_size=figsize , cmap=cmap_train , normalize=normalize , colorbar=colorbar ,label='Testing Data' ,target_names=target_names)
    if output_dict == True :
        results_dit = {'train' : result_train , 'test' : result_test}
        print("Trainning Metrics")
        display(pd.DataFrame(results_dit['train']).round(2))
        print()
        print("Testing Metrics")
        display(pd.DataFrame(results_dit['test']).round(2))
   
        
def convert_probs_to_preds(probs , threshold ,pos_class = 1):
    predictions = [1 if prob[pos_class] > threshold else 0 for prob in probs]
    return predictions

def plot_threshold_metrics(model, X_test, y_test, 
                           start=0, stop=1.05, step=0.05,
                           pos_class=1, figsize=(15,5)):
    # Get probabilities
    y_probs = model.predict_proba(X_test)
    
    # Generate thresholds
    thresholds = np.arange(start=start, stop=stop, step=step)
    
    # Empty lists
    recalls     = []
    precisions  = []
    accuracies  = []

    # Iterate over thresholds
    for thresh in thresholds:
        preds = convert_probs_to_preds(y_probs, thresh, pos_class=pos_class)
        recalls.append(recall_score(y_test, preds))
        precisions.append(precision_score(y_test, preds))
        accuracies.append(accuracy_score(y_test, preds))

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(thresholds, recalls,    label='Recall')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.legend()
    plt.title('Precision, Recall, and Accuracy Scores Across Decision Thresholds')
    plt.xlabel('Decision Thresholds')
    plt.ylabel('Score')
    plt.grid()
    plt.xticks(thresholds, rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Return as dataframe for reference
    results_df = pd.DataFrame({
        'Threshold' : thresholds,
        'Recall'    : recalls,
        'Precision' : precisions,
        'Accuracy'  : accuracies
    }).round(3)
    
    return results_df

def evaluate_with_threshold(model, X_train, y_train, X_test, y_test,
                             threshold=0.3, pos_class=1,
                             target_names=None,
                             figsize=(8,4), normalize='true',
                             cmap_train='Blues', cmap_test='Reds',
                             colorbar=False, output_dict=False):
    """
    Evaluates a model using a custom decision threshold.
    Shows classification metrics for both training and testing data.

    Parameters:
    -----------
    model        : fitted model with predict_proba method
    X_train      : training features
    y_train      : training labels
    X_test       : testing features
    y_test       : testing labels
    threshold    : decision threshold (default 0.3)
    pos_class    : positive class index (default 1)
    target_names : list of class names (default None)
    figsize      : figure size (default (8,4))
    normalize    : confusion matrix normalization (default 'true')
    cmap_train   : colormap for training matrix (default 'Blues')
    cmap_test    : colormap for testing matrix  (default 'Reds')
    colorbar     : show colorbar (default False)
    output_dict  : return metrics as dict (default False)
    """
    print(f"{'='*70}")
    print(f"  Threshold = {threshold}")
    print(f"{'='*70}\n")

    # Step 1 — Get probabilities
    y_train_probs = model.predict_proba(X_train)
    y_test_probs  = model.predict_proba(X_test)

    # Step 2 — Convert with threshold
    y_train_pred = convert_probs_to_preds(y_train_probs,
                                          threshold=threshold,
                                          pos_class=pos_class)
    y_test_pred  = convert_probs_to_preds(y_test_probs,
                                          threshold=threshold,
                                          pos_class=pos_class)

    # Step 3 — Show metrics
    result_train = classification_metrics(y_train, y_train_pred,
                                          target_names=target_names,
                                          label='Training Data',
                                          fig_size=figsize,
                                          normalize=normalize,
                                          cmap=cmap_train,
                                          colorbar=colorbar,
                                          output_dict=output_dict)
    print()
    result_test = classification_metrics(y_test, y_test_pred,
                                         target_names=target_names,
                                         label='Testing Data',
                                         fig_size=figsize,
                                         normalize=normalize,
                                         cmap=cmap_test,
                                         colorbar=colorbar,
                                         output_dict=output_dict)

    # Step 4 — Return dict if needed
    if output_dict:
        results = {'train': result_train, 'test': result_test}
        print("Training Metrics")
        display(pd.DataFrame(results['train']).round(2))
        print()
        print("Testing Metrics")
        display(pd.DataFrame(results['test']).round(2))
        return results