import pandas as pd
import datetime as dt
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import LinearSVC
from sklearn import metrics

#own imports - switch to importing all methods at once
from data_handler.Data_loader import *
from data_handler.OFF_data_loader import *
from data_handler.Data_balancer import *
from data_cleaning import *
#from visualization.roc_curve import plot_multiclass_roc

"""
Main file in which all the steps of the model are called in a successive order
"""

if __name__ == "__main__":

    #set the working directory
    path = r"C:\Users\Giorgio\Desktop\ETH\Code"
    os.chdir(path)

    # Define locations
    saveLoc = '/output/classification_reports/'
    ReportName = str('classification_report-' + dt.datetime.now().strftime('%Y-%m-%d-%H-%M') + '.txt')
    class_report_path = saveLoc + ReportName

    # Load the parameters file
    if os.path.exists(os.getcwd() +'\Build\model_params.yml'):
        with open(os.getcwd() +'\Build\model_params.yml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print('Parameters file is missing.')

    if params['ReloadData'] is True:
        language = params['Language']

        if params['Database'] == 'Eatfit':
            column = 'text'
            # Load data from the Eatfit SQL database
            df = query_eatfit_db_ingr_ubp(language)

        if params['Database'] == 'OpenFoodFacts':
            # Load data from the OFF mongodb database
            DOMAIN = 'localhost'
            PORT = 27017
            column = 'ingredients_text_{}'.format(params['Language'])
            OFF_db = connect_mongo(DOMAIN, PORT)
            if language =='de':
                df = query_off_data_de(OFF_db)
            elif language == 'en':
                df = query_off_data_en(OFF_db)
            else:
                None

        print('Data retrieved')
        print(df.head())

        # Drop rows without an ingredients list
        if params['Database']=='Eatfit':
            df = clean_dataframe(df, params['Language'])
        else:
            df = clean_OFF_dataframe(df, params['Language'])

        # #check for empty values
        # df = check_for_NaN_values(df)

        # Print a summary of the data
        # text= eatfit_data_summary(df)
        # print(text)

        # Clean the ingredient list text
        cleaned_dt = text_cleaning(df, params['Language'], column)
        print(cleaned_dt.head())

        # save interim results as csv file
        # cleaned_dt.to_csv(os.getcwd() +'/interim_results/cleaned_data.csv')
        if params['Database']=='Eatfit':
            cleaned_dt.to_pickle(os.getcwd() +'/interim_results/cleaned_data.pkl')
        else:
            cleaned_dt.to_pickle(os.getcwd() +'/interim_results/cleaned_data_OFF.pkl')
        # another interesting option would be to_parquet

    else:
        if params['Database']=='Eatfit':
            cleaned_dt = pd.read_pickle(os.getcwd() +'/interim_results/cleaned_data.pkl')
        else:
            cleaned_dt = pd.read_pickle(os.getcwd() +'/interim_results/cleaned_data_OFF.pkl')
        # cleaned_dt = pd.read_csv(os.getcwd() +'/interim_results/cleaned_data.csv')

    # ------------------MODEL-------------------------

    # Split the data into train & test sets

    if params['Database']=='Eatfit':
        X = cleaned_dt['text']
        y = cleaned_dt['ubp_score']
    else:
        X = cleaned_dt['ingredients_text_{}'.format(params['Language'])]
        y = cleaned_dt['ecoscore_grade']

    if params['DataBalancing']== 'RandomUpsampling':
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X)
        X, y = random_upsampler(X,y) # ,random_state=0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['SplitSize'], random_state=params['RandomState'])

    # Build a pipeline, train and fit the model

    if params['classifier'] == "RandomForest":
        clf = RandomForestClassifier()
    elif params['classifier'] == "NaiveBayes":
        clf = MultinomialNB()

    if params['DataBalancing']== 'RandomUpsampling':
        estimators = [('clf', clf),]
    else:
        estimators = [('tfidf', TfidfVectorizer()), ('clf', clf),]

    text_clf = Pipeline(estimators) # some preprocessing could be avoided by adding few parameters in the model.

    text_clf.fit(X_train, y_train)

    # Form a prediction set
    predictions = text_clf.predict(X_test)

    # --------------RESULTS------------------
    # Report the confusion matrix, the classification report, and the  overall accuracy
 
    txt_block = [
        str("Date: " + dt.datetime.now().strftime('%d/%m/%Y %H:%M')),
        str("Database: " + params['Database']),
        str("Language: " + params['Language']),
        str("Classifier:" + params['classifier']),
        "Split Size: " + str(params['SplitSize']), '\n',
        "CONFUSION MATRIX =",
        metrics.confusion_matrix(y_test,predictions), '\n',
        "CLASSIFICATION REPORT =",
        metrics.classification_report(y_test,predictions), '\n',
        "ACCURACY SCORE =", 
        metrics.accuracy_score(y_test,predictions), '\n'
     ]

    with open(os.getcwd() + class_report_path, 'w') as f:
        for txt in txt_block:
            f.write(str(txt))
            f.write('\n')

    print("Completed successfully")
    
    # print(metrics.confusion_matrix(y_test,predictions))
    # print(metrics.classification_report(y_test,predictions))
    # print(metrics.accuracy_score(y_test,predictions))

    # generate the ROC curve
    # plot_multiclass_roc()
    # clf = OneVsRestClassifier(RandomForestClassifier()).fit(X, y)