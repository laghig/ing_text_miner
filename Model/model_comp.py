import datetime as dt
from re import I
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
from sklearn.model_selection import cross_validate, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from visualization.class_plots import plot_variance_explained

# Own imports
from visualization.roc_curve import roc_curve
from data_handler.Data_balancer import *
from visualization.reg_plots import *
from visualization.roc_curve import *
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
        """
        Function to build the model pipeline, fit the model and perform hyperparameters tuning
        """

        # Assembling the pipeline
        estimators=[]

        feature_extraction = {
            'tf-idf': TfidfVectorizer(),
            'count_vec': CountVectorizer()
        }
        estimators.append(('feat_ext', feature_extraction[self.modelparams['feature_ext']]))

        if self.modelparams['transformation'] == 'pca':
            estimators.append(('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)))

        transformations = {
            'var_thres': VarianceThreshold(),
            'tr_svd': TruncatedSVD(n_components=self.modelparams['num_components']),
            'pca': PCA(n_components=self.modelparams['num_components']),
            'trf': FunctionTransformer(lambda x: x.toarray(), accept_sparse=True) # convert sparse into dense matrix
        }
        if self.modelparams['transformation'] is not None:
            estimators.append(('trsf', transformations[self.modelparams['transformation']]))

        oversampler = {
            'RandomUpsampling': RandomOverSampler(sampling_strategy='minority'),
            'smote': SMOTE(k_neighbors=3),

        }
        if self.modelparams['DataBalancing']== True:
            estimators.append(('over_smplr', oversampler[self.modelparams['Balancer']]))

        classifiers = {
            "RandomForest": RandomForestClassifier(),
            "NaiveBayes": MultinomialNB(alpha=0.1),
            "KNN": KNeighborsClassifier(leaf_size=6, n_neighbors=3, p=2),
            "DummyClassifier": DummyClassifier()
        }
        if self.modelparams['approach']=='classification':
            estimators.append(('clf', classifiers[self.modelparams['algorithm']]))

        regressors = {
            'ridge': Ridge(alpha=0.5, positive=True),
            "KNN": KNeighborsRegressor(leaf_size=5, n_neighbors=4, p=1),
            'lasso': Lasso(alpha=0.0004, positive=True)
        }
        if self.modelparams['approach']=='regression':
            estimators.append(('clf', regressors[self.modelparams['algorithm']]))

        # Split the dataset in train and test samples
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.modelparams['SplitSize'], random_state=self.modelparams['RandomState'])

        # Build a Pipeline
        self.text_clf = Pipeline(estimators) # some preprocessing could be avoided by adding few parameters in the model.

        # data_balancing_summary(self.y, dict(self.text_clf.steps).get('sampler').sampling_strategy_)

        if self.modelparams['Hyperparameter_opt']== True:
            hyperparameters= get_hyperparameters(self.modelparams['algorithm'])
            print(hyperparameters)
            score = self.modelparams['Hyperparameter_score']
            #Use GridSearch with the previously defined classifier
            self.text_clf = GridSearchCV(self.text_clf, hyperparameters, scoring=score, cv=5)
            
        #Fit the model
        self.text_clf.fit(self.X_train, self.y_train)

        #Print The value of best Hyperparameters
        if self.modelparams['Hyperparameter_opt']== True:
                grid_search_results = pd.DataFrame(self.text_clf.cv_results_)
                grid_search_results.to_csv(r"C:\Users\Giorgio\Desktop\best_parameter_multinomialnb.csv")
                print(grid_search_results)

        # Form a prediction set
        self.predictions = self.text_clf.predict(self.X_test)


    def visualize(self):
        """
        Generate the plots specific to each model
        """
        if self.modelparams['algorithm']=='ridge':
            reg_coeff = self.text_clf['clf'].coef_.tolist()
            print("total # of oefficient: " + str(len(reg_coeff))),
            print("non-zero coefficients: " + str(np.count_nonzero(reg_coeff)))
            feature_dict = self.text_clf['feat_ext'].vocabulary_
            ordered_features = dict(sorted(feature_dict.items(), key=lambda item: item[1]))
            labeled_coeff = list(merge(ordered_features.keys(), reg_coeff))
            labeled_coeff.sort(key=lambda i:i[1],reverse=True)
            plot_reg_coeff(labeled_coeff[:30])

        if self.modelparams['transformation']=='pca':
            # Print a plot of explained variance against number of components
            variance_explained = np.cumsum(self.text_clf['trsf'].explained_variance_)
            plot_variance_explained(self.modelparams['num_components'], variance_explained, self.modelparams['algorithm'])
        
        if self.modelparams['approach']=='classification':
            i = 2 if self.modelparams['DataBalancing'] is True else 1
            classes = ['A', 'B', 'C', 'D', 'E']
            # roc_curve_comp(self.text_clf[:-i], self.X, self.y, classes, self.modelparams['DataBalancing'])
            multiclass_roc_curve(self.text_clf[:-i],self.modelparams['algorithm'], self.X, self.y, classes, self.modelparams['DataBalancing'], 'single')
        

    
    def report(self):
        """
        Generate a text report with cross-validated results
        """
        vectorized_X = TfidfVectorizer().fit_transform(self.X_train)

        # Cross validation
        if self.modelparams['approach']=='classification':
            scoring = ['precision_macro', 'recall_macro', 'accuracy','f1_macro']
        elif self.modelparams['approach']=='regression':
            scoring= ['r2', 'neg_mean_squared_error']
        cv =  RepeatedKFold(n_splits=5, n_repeats=5, random_state=self.modelparams['RandomState'])
        self.scores = cross_validate(self.text_clf, self.X, self.y, cv=cv, scoring=scoring)
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
                'Almost Correct Predictions Error Rate: ' + str(sum(acper(self.y_test,self.predictions))/len(self.y_test)), '\n',
                "Pearson correlation coeff.:" + str(scipy.stats.pearsonr(self.y_test,self.predictions)),
                "Spearman correlation coeff.:" + str(scipy.stats.spearmanr(self.y_test,self.predictions)),
             ]