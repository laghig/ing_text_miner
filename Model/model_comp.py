import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from statistics import mean
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline 
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA,  TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor
# from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from collections import Counter
# from main import X


#Own imports
#from model_mod import *
from visualization.roc_curve import roc_curve
from data_handler.Data_balancer import *
from visualization.reg_plots import *
from Model.utils import *

class ModelStructure:
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()
    prediction = pd.DataFrame()
    dummy_prediction = pd.DataFrame()
    scores = 0
    dummy_scores = 0
    txt_block = ""
    trsf= ""

    def __init__(self, X, y, modelparams, modifications):
        self.X = X
        self.y=y
        self.modelparams = modelparams
        self.modifications = modifications
        self.text_clf = Pipeline

        # self.assemble()
        # self.report()

    def assemble(self):

        # init_class_distribution = Counter(self.y)
        # print(init_class_distribution)
        # min_class_label, _ = init_class_distribution.most_common()[-4]

        # Feauture extraction method
        if self.modelparams['feature_ext']=='tf-idf':
            trsf = TfidfVectorizer()
        else:
            trsf = CountVectorizer()

        # Additional transformations
        trf = None
        pca = None
        NUM_COMPONENTS = 1800 # 3575

        # Select an algorithm
        if self.modelparams['approach']=='classification':
            if self.modelparams['algorithm'] == "RandomForest":
                clf = RandomForestClassifier()
                # trf = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)
                # pca = PCA(n_components=NUM_COMPONENTS)
                pca = TruncatedSVD(n_components=NUM_COMPONENTS, random_state=42)
                pca = VarianceThreshold()
            elif self.modelparams['algorithm'] == "NaiveBayes":
                clf = MultinomialNB()
            elif self.modelparams['algorithm'] == "KNN":
                clf = KNeighborsClassifier()
            elif self.modelparams['algorithm'] == "DummyClassifier":
                clf = DummyClassifier()
        if self.modelparams['approach']=='regression':
            if self.modelparams['algorithm']=='ridge':
                clf = Ridge(alpha=0.5, positive=True) #alpha=0.5, positive=True
            elif self.modelparams['algorithm']=='lasso':
                clf = Lasso(alpha=0.0004, positive=True) # alpha=0.001
            elif self.modelparams['algorithm'] == "KNN":
                clf = KNeighborsRegressor(leaf_size=5, n_neighbors=4, p=1) #leaf_size=1, p=2, n_neighbors=4

        
        # Data balancing
        over_smplr = None
        if self.modelparams['DataBalancing']== True:
            if self.modelparams['Balancer']== 'RandomUpsampling':
                over_smplr = RandomOverSampler(sampling_strategy='minority') # ,random_state=0
            elif self.modelparams['Balancer']== 'smote':
                over_smplr = SMOTE() # sampling_strategy = {'A': 800, 'B':970, 'C': 1393, 'D':847, 'E':1196}
            elif self.modelparams['Balancer']== 'adasyn':
                over_smplr = ADASYN()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.modelparams['SplitSize'], random_state=self.modelparams['RandomState'])

        # Build a Pipeline
        estimators = [('tfidf', trsf), ('trf', trf) , ('pca', pca), ('sampler', over_smplr), ('clf', clf)]
        # estimators = [('cv', CountVectorizer()), ('clf', clf),]
        self.text_clf = Pipeline(estimators) # some preprocessing could be avoided by adding few parameters in the model.



        
        if self.modelparams['Hyperparameter_opt']== True:
            if self.modelparams['algorithm'] == "KNN":
                leaf_size = list(range(1,10))
                n_neighbors = list(range(1,10))
                p=[1,2]
                hyperparameters = dict(clf__leaf_size=leaf_size, clf__n_neighbors=n_neighbors, clf__p=p)
                score = 'neg_mean_squared_error'
            elif self.modelparams['algorithm']== "NaiveBayes":
                hyperparameters = {

                }
            elif self.modelparams['algorithm'] == "ridge" or "lasso":
                alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]# Ridge: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 1]
                t_f = [True, False]
                hyperparameters = dict(clf__alpha = alphas ) #clf__positive=t_f
                score = 'neg_mean_squared_error'  # 'neg_mean_squared_error' / 'r2'

            #Use GridSearch with the previously defined classifier
            self.text_clf = GridSearchCV(self.text_clf, hyperparameters, scoring=score, cv=5)
            #Fit the model
        

        # dummy_clf.fit(self.X_train, self.y_train)
        self.text_clf.fit(self.X_train, self.y_train)

        # sampling_strategy = dict(self.text_clf.steps).get('sampler').sampling_strategy_
        # expected_n_samples = sampling_strategy.get(min_class_label)
        # print(f'Expected number of generated samples: {expected_n_samples}')

        #Print The value of best Hyperparameters
        if self.modelparams['Hyperparameter_opt']== True:
                grid_search_results = pd.DataFrame(self.text_clf.cv_results_)
                # grid_search_results.to_csv(r"C:\Users\Giorgio\Desktop\best_parameter.csv")
                print(grid_search_results)
            # if self.modelparams['algorithm']== "KNN":
            #     print('Best leaf_size:', self.text_clf.best_estimator_.get_params()['leaf_size'])
            #     print('Best p:', self.text_clf.best_estimator_.get_params()['p'])
            #     print('Best n_neighbors:', self.text_clf.best_estimator_.get_params()['n_neighbors'])
            # elif self.modelparams['algorithm']== "NaiveBayes":
            #     print('Best alpha: ')
            # elif self.modelparams['algorithm']== "ridge" or "lasso":
            #     # print(self.text_clf.best_estimator_.get_params())
            #     grid_search_results = pd.DataFrame(self.text_clf.cv_results_)
            #     print(grid_search_results)
            #     # params = pd.DataFrame([i[0] for i in self.text_clf.grid_scores_])
            #     # results = pd.DataFrame(self.text_clf.grid_scores_)
            #     # results = pd.concat([params, results], 1)
            #     # results["rmse"] = np.sqrt(-results.mean_validation_score)
            #     # print(results.head(9))



        # Form a prediction set
        # self.dummy_prediction = dummy_clf.predict(self.X_test)
        self.predictions = self.text_clf.predict(self.X_test)

        # Print a plot of explained variance against number of components
        # variance_explained = np.cumsum(pca.explained_variance_)
        # fig, ax = plt.subplots(figsize=(15, 8))
        # plt.plot(range(NUM_COMPONENTS),variance_explained, color='tab:blue')
        # ax.grid(True)
        # plt.xlabel("Number of components", fontsize=18)
        # plt.ylabel("Cumulative explained variance", fontsize=18)
        # plt.tick_params(labelsize=14)
        # plt.tight_layout()
        # # plt.savefig(r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots\poc_components.png")
        # plt.show()


    def visualize(self):
        if self.modelparams['algorithm']=='ridge':
            reg_coeff = self.text_clf['clf'].coef_.tolist()
            print("total # of oefficient: " + str(len(reg_coeff))),
            print("non-zero coefficients: " + str(np.count_nonzero(reg_coeff)))
            feature_dict = self.text_clf['tfidf'].vocabulary_
            ordered_features = dict(sorted(feature_dict.items(), key=lambda item: item[1]))
            labeled_coeff = list(merge(ordered_features.keys(), reg_coeff))
            labeled_coeff.sort(key=lambda i:i[1],reverse=True)
            plot_reg_coeff(labeled_coeff[:30])
        

    
    def report(self):
        vectorized_X = TfidfVectorizer().fit_transform(self.X_train)

        # Cross validation
        if self.modelparams['approach']=='classification':
            scoring = ['precision_macro', 'recall_macro', 'accuracy','f1_macro']
        elif self.modelparams['approach']=='regression':
            scoring= ['r2', 'neg_mean_squared_error']
        cv =  RepeatedKFold(n_splits=5, n_repeats=5, random_state=self.modelparams['RandomState'])
        self.scores = cross_validate(self.text_clf, self.X, self.y, cv=cv, scoring=scoring)
        # self.dummy_scores = cross_validate(dummy_clf, self.X, self.y, cv=5, scoring=scoring)
        # print(self.scores.keys())


        # Assemble the report layout
        if self.modelparams['approach'] == "classification":

            self.txt_block = [
                    str("Classifier:" + self.modelparams['algorithm']),
                    # str("Text feauture extractor:" + params['FeautureExt']),
                    "Split Size: " + str(self.modelparams['SplitSize']),
                    "X_train shape: " + str(vectorized_X.shape),
                    "y_train shape: " + str(self.y_train.shape), 
                    "Data balancing: " + str(self.modelparams['DataBalancing']) + ', with '+ str(self.modelparams['Balancer']),'\n',
                    "1. SINGLE RUN",
                    "CONFUSION MATRIX =",
                    metrics.confusion_matrix(self.y_test,self.predictions), '\n',
                    # metrics.confusion_matrix(self.y_test,self.dummy_prediction), '\n',
                    "CLASSIFICATION REPORT =",
                    metrics.classification_report(self.y_test,self.predictions), '\n',
                    # metrics.classification_report(self.y_test,self.dummy_prediction), '\n',
                    "ACCURACY SCORE =", 
                    metrics.accuracy_score(self.y_test,self.predictions), '\n'
                    "MCC =", metrics.matthews_corrcoef(self.y_test,self.predictions), '\n'
                    # "2. Cross validation (5-fold): ",
                    "Accuracy: {} with a standard deviation of {}".format(round(self.scores['test_accuracy'].mean(),3), round(self.scores['test_accuracy'].std(),3)),
                    "Precision: {} with a standard deviation of {}".format(round(self.scores['test_precision_macro'].mean(),3), round(self.scores['test_precision_macro'].std(),3)),
                    "Recall: {} with a standard deviation of {}".format(round(self.scores['test_recall_macro'].mean(),3), round(self.scores['test_recall_macro'].std(),3)),
                    "F1-Macro: {} with a standard deviation of {}".format(round(self.scores['test_f1_macro'].mean(),3), round(self.scores['test_f1_macro'].std(),3)),
             ]
        


        elif self.modelparams['approach'] == 'regression':

            self.txt_block = [
                str('Method: ' + self.modelparams['algorithm'] + " " + self.modelparams['approach']),
                "Split Size: " + str(self.modelparams['SplitSize']), '\n',
                "X_train shape: " + str(vectorized_X.shape),
                "y_train shape: " + str(self.y_train.shape), 
                "avg number of ingredients: " + str(vectorized_X.count_nonzero()/len(vectorized_X.toarray())), '\n',
                "R-squared (single run): " + str(metrics.r2_score(self.y_test, self.predictions)),
                # 'Cross validation (5-fold): ' + str(self.scores['r2']),
                "R-squared: %0.2f mean with a standard deviation of %0.2f" % (round(self.scores['test_r2'].mean(),3), round(self.scores['test_r2'].std(),3)), '\n',
                "Mean squared error: {} with a standard deviation of {}".format(round(self.scores['test_neg_mean_squared_error'].mean(),3), round(self.scores['test_neg_mean_squared_error'].std(),3)), '\n',
                # "Mean squared error: " + str(metrics.mean_squared_error(self.y_test, self.predictions)), '\n',                
                "Pearson correlation coeff.:" + str(scipy.stats.pearsonr(self.y_test,self.predictions)),
                "Spearman correlation coeff.:" + str(scipy.stats.spearmanr(self.y_test,self.predictions)),
             ]
        

    # def visualize(self):

    #     plt.figure()
    #     models = [self.dummy_clf, self.text_clf]
    #     for mod in models:
    #         y_pred = mod.predict_proba(self.X_test)[:, 1]
    #         fpr, tpr, _ = metrics.roc_curve(self.y_test, y_pred)
    #         auc = round(metrics.roc_auc_score(self.y_test, y_pred), 4)
    #         plt.plot(fpr,tpr,label= str(mod) + ", AUC="+str(auc))

    #     lw=1
    #     plt.plot([0, 1], [0, 1], "k--", lw=lw)
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.title("Multiclass ROC curve", weight='bold')
    #     plt.legend(loc="lower right")
    #     #plt.savefig(r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots\roc_curve.jpg") # uncomment to save the plot
    #     plt.show


if __name__ == "__main__":
    print('hello')