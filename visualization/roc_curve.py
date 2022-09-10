import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

cwd = os.getcwd()

### Classifiers
classifiers = {
    'RandomForest': OneVsRestClassifier(RandomForestClassifier()),
    'NaiveBayes': OneVsRestClassifier(MultinomialNB()),
    'KNN': OneVsRestClassifier(KNeighborsClassifier()),
    }

def assemble_clf(classifier, data_balancing):
    clf = classifiers[classifier]
    if data_balancing == True:
        clf = Pipeline([('sampl', SMOTE()), 
                        ('clf', clf)])

    ovr = OneVsRestClassifier(clf)
    return ovr



def multiclass_roc_curve(text_clf, model, X, y, classes, data_balancing,  disp): # other input parameters: clf, figsize, n_classes
    """
    This function return a OnevsRest roc curve for a specified model
    :param text_clf: pre-processing pipeline (without classifier)
    :model: classifier
    :X,y: dataset with the dependent and independend variables
    :classes: List of the different class labels
    :disp: parameters to plot single classes roc curve ('single') or a class averaged roc curve ('avg')
    :returns: a matplotlib plot
    """
    file_name = "\output\plots\ROC_curve_{}.png".format(model)
    # Binarize the output
    y = label_binarize(y, classes=classes)
    n_classes = y.shape[1]

    X = text_clf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    ovr = assemble_clf(model, data_balancing)
    y_score = ovr.fit(X_train, y_train).predict_proba(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and ROC area for each class

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    lw=1

    # Plot all ROC curves
    plt.figure()
    if disp == 'avg':
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color="firebrick",
            linestyle="dashdot",
            linewidth=lw,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
            color="black",
            linestyle="dashdot",
            linewidth=lw,
        )

    dict_class = {
        '0': 'A',
        '1': 'B',
        '2': 'C',
        '3': 'D',
        '4': 'E'
    }
    if disp == 'single':
        colors = cycle(["steelblue", "seagreen", "black","firebrick", "grey"])
        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label="ROC curve of class {0} (area = {1:0.2f})".format(dict_class[str(i)], roc_auc[i]),
            )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC curve", weight='bold')
    plt.legend(loc="lower right")
    plt.savefig(cwd + file_name) # uncomment to save the plot
    plt.show()

def roc_curve_comp(text_clf, X, y, classes, data_balancing):
    """
    This function visualize the macro-average roc curve of different models in one single plot
    :param text_clf: pre-processing pipeline (without classifier)
    :X,y: dataset with the dependent and independend variables
    :classes: List of the different class labels
    :returns: a matplotlib plot
    """
    file_name = "\output\plots\ROC_curve_model_ comparison.png"
    y = label_binarize(y, classes=classes)
    n_classes = y.shape[1]

    X = text_clf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for clf in classifiers:
        ovr = assemble_clf(clf, data_balancing)
        y_score = ovr.fit(X_train, y_train).predict_proba(X_test)

        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro_{}".format(clf)] = all_fpr
        tpr["macro_{}".format(clf)] = mean_tpr
        roc_auc["macro_{}".format(clf)] = round(auc(fpr["macro_{}".format(clf)], tpr["macro_{}".format(clf)]),2)
        lw=1

    layout= {
        'RandomForest': 'black',
        'NaiveBayes': "firebrick",
        'k-NearestNeighbor': 'peru',
        'Dummy': 'grey' 
    }

    # Plot all ROC curves
    plt.figure()

    for clf in classifiers:
        plt.plot(
            fpr["macro_{}".format(clf)],
            tpr["macro_{}".format(clf)],
            label="{} ROC curve (auc = {})".format(clf, roc_auc["macro_{}".format(clf)]), # 0:0.2f
            color=layout[clf],
            # linestyle="dashdot",
            linewidth=lw,
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC curve", weight='bold')
    plt.legend(loc="lower right")
    plt.savefig(cwd + file_name) # uncomment to save the plot
    plt.show()


if __name__ == "__main__":
    # import interim data as a pandas df
    df = pd.read_pickle(r"C:\Users\Giorgio\Desktop\ETH\Code\interim_results\cleaned_data.pkl")
    classes = ['A', 'B', 'C', 'D', 'E'] # [1, 2, 3]
    disp = 'avg' # 'single' / 'avg'

    # train test split
    
    X = df['text']
    y = df['ubp_score']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    multiclass_roc_curve(X,y, classes, disp)