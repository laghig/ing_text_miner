import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from pip import main
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC
from sklearn import metrics

from Data_loader import query_eatfit_db, check_for_NaN_values, eatfit_data_summary
from data_cleaning import data_cleaning

"""
Main file in which all the steps of the model are called in a successive order
"""

if __name__ == "__main__":
    
    # Load data from the Eatfit SQL database
    df = query_eatfit_db(query='nutri_score_ingr_en')
    print(df.head())

    # Load data from the OFF mongodb database

    # Drop empty values
    df.dropna(inplace=True)

    # Delete other data
    # df = df[df.text != 'Product information’s are not available in English']
    df = df[df.text != '-']

    #check for empty values
    df = check_for_NaN_values(df)

    # Print a summary of the data
    text= eatfit_data_summary(df)
    print(text)

    # Clean the ingredient list text
    cleaned_dt = data_cleaning(df)
    print(cleaned_dt.head())

    # ------------------MODEL-------------------------

    # Split the data into train & test sets

    X = df['text']
    y = df['BLS_Code']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Build a pipeline, train and fit the model

    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier()),]) # some preprocessing could be avoided by adding few parameters in the model

    text_clf.fit(X_train, y_train)

    # Form a prediction set
    predictions = text_clf.predict(X_test)

    # Report the confusion matrix, the classification report, and the  overall accuracy

    print(metrics.confusion_matrix(y_test,predictions))

    print(metrics.classification_report(y_test,predictions))

    print(metrics.accuracy_score(y_test,predictions))



    

