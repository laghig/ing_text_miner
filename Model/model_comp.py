import datetime as dt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
# from sklearn.svm import LinearSVC
from sklearn import metrics
# from main import X


#Own imports
#from model_modifications import *

class ModelStructure:
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()
    prediction = pd.DataFrame()
    txt_block = ""

    def __init__(self, X, y, modelparams):
        self.X = X
        self.y=y
        self.modelparams = modelparams

        # self.assemble()
        # self.report()

    def assemble(self):

        # Selecting a classifier
        if self.modelparams['approach']=='classification':
            if self.modelparams['classifier'] == "RandomForest":
                clf = RandomForestClassifier()
            elif self.modelparams['classifier'] == "NaiveBayes":
                clf = MultinomialNB()
        if self.modelparams['approach']=='linearReg':
            clf = Ridge()
        
        # Build a Pipeline
        if self.modelparams['BalancedData']== True:
            estimators = [('clf', clf),]
        else:
            estimators = [('tfidf', TfidfVectorizer()), ('clf', clf),]
            # estimators = [('cv', CountVectorizer()), ('clf', clf),]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.modelparams['SplitSize'], random_state=self.modelparams['RandomState'])

        text_clf = Pipeline(estimators) # some preprocessing could be avoided by adding few parameters in the model.

        text_clf.fit(self.X_train, self.y_train)

        # Form a prediction set
        self.predictions = text_clf.predict(self.X_test)
    
    def report(self):
        if self.modelparams['approach'] == "classification":
            self.txt_block = [
                    str("Classifier:" + self.modelparams['classifier']),
                    # str("Text feauture extractor:" + params['FeautureExt']),
                    "Split Size: " + str(self.modelparams['SplitSize']), '\n',
                    "CONFUSION MATRIX =",
                    metrics.confusion_matrix(self.y_test,self.predictions), '\n',
                    "CLASSIFICATION REPORT =",
                    metrics.classification_report(self.y_test,self.predictions), '\n',
                    "ACCURACY SCORE =", 
                    metrics.accuracy_score(self.y_test,self.predictions), '\n'
                    "MCC =", metrics.matthews_corrcoef(self.y_test,self.predictions), '\n'
             ]

        if self.modelparams['approach'] == 'linearReg':
            self.txt_block = [
                "R squared: " + str(metrics.r2_score(self.y_test, self.predictions)),
                "Mean squared error: " + str(metrics.mean_squared_error(self.y_test, self.predictions)), '\n',
             ]



