import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


def multiclass_roc_curve(X, y): # other input parameters: clf, figsize, n_classes
    # Binarize the output
    y = label_binarize(y, classes=['A', 'B', 'C', 'D', 'E'])
    n_classes = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    ovr = OneVsRestClassifier(RandomForestClassifier())
    # ovr.fit(X, y_test)
    # yhat = ovr.predict(X)
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
    # plt.plot(
    #     fpr["micro"],
    #     tpr["micro"],
    #     label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    #     color="firebrick",
    #     linestyle="dashdot",
    #     linewidth=lw,
    # )

    # plt.plot(
    #     fpr["macro"],
    #     tpr["macro"],
    #     label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    #     color="black",
    #     linestyle="dashdot",
    #     linewidth=lw,
    # )

    dict_class = {
        '0': 'A',
        '1': 'B',
        '2': 'C',
        '3': 'D',
        '4': 'E'
    }

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
    #plt.savefig(r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots\roc_curve.jpg") # uncomment to save the plot
    plt.show()

if __name__ == "__main__":
    # import interim data as a pandas df
    df = pd.read_csv(r"C:\Users\Giorgio\Desktop\ETH\Code\interim_results\cleaned_data.csv")

    # train test split
    
    X = df['text']
    y = df['ubp_score']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    multiclass_roc_curve(X,y)