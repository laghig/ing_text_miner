import string
import MySQLdb
import mysql.connector
from mysql.connector import Error
import yaml
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC
from sklearn import metrics
import sys
import os


#set the working directory
path = r"C:\Users\Giorgio\Desktop\ETH\Code"
os.chdir(path)

# Load the configuration file
if os.path.exists("config.yml"):
    with open(os.getcwd() +'\config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
else:
    print('Config file is missing.')

# function to fetch data from the mysql database 
"""
Input: sql query as a string
Output: pandas dataframe
"""
def query_eatfit_db(sql_query=string):
    try:
        connection = mysql.connector.connect(host=config['mysql_db']['Host'],
                            user=config['mysql_db']['User'],
                            passwd=config['mysql_db']['Password'],
                            db=config['mysql_db']['Database'])

        sql_select_Query = sql_query

        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to MySQL Server version ", db_Info)
            cursor = connection.cursor()

            # SQL queries
            cursor.execute(sql_select_Query)
            records = cursor.fetchall()
            ls = []
            for row in records:
                ls.append(row)
            df = pd.DataFrame(ls, columns=[i[0] for i in cursor.description])
            return df    

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

def data_cleaning(df):

    # Stop words
    en_stopwords = stopwords.words('english')
    de_stopwords = stopwords.words('german')
    # add few words to the stop words dictionary
    en_stopwords.extend(['ingredients'])

    # Use Porter stemmer
    stemmer = PorterStemmer()

    df1= pd.DataFrame(columns = ['text', 'id', 'nutri_score_calculated'])

    for index, row in df.iterrows():
        ingr_ls = row['text'].translate(str.maketrans('', '', string.punctuation))
        ingr_ls = word_tokenize(ingr_ls)
        filtered_words = [word.lower() for word in ingr_ls if word not in en_stopwords]
        ingr_ls_stem = [stemmer.stem(word) for word in filtered_words]

        ingr_clean = [' '.join( ingr for ingr in ingr_ls_stem)]

        ingr_clean.append(row['id'])
        ingr_clean.append(row['nutri_score_calculated'])

        new_row = pd.Series(ingr_clean, index = df1.columns)
        df1 = df1.append(new_row, ignore_index=True)

    return df1

# QUERIES

# Query all major categories
maj_categories = "SELECT * FROM nutritiondb.major_category"

# Query a specific nutritional table
prod_id = 16
nutr_table =    "SELECT  name, amount, unit_of_measure, is_mixed \
                FROM nutritiondb.nutrition_fact \
                WHERE product_id = {};".format(prod_id)

# Query all ingredients lists
all_ingr_ls = "SELECT * \
            FROM nutritiondb.ingredient"

# Query all english ingredients lists
all_en_ingr_ls = "SELECT * \
                FROM nutritiondb.ingredient \
                WHERE lang = 'en' "

# Query the ingredients list for a specific product
prod_id = 16
ingr_list=  "SELECT lang, text \
            FROM nutritiondb.ingredient \
            WHERE product_id = {};".format(prod_id)
# in this case would not be better to generate a dictionary, with the language a key and the ingridients list as a value?

# Query to retrieve the ingredient lists with the respective nutri-score
nutri_score_ingr_en = "SELECT  product.id, a.text, a.lang, product.nutri_score_calculated \
                        FROM  nutritiondb.product \
                        LEFT JOIN \
                        (SELECT * \
                        FROM nutritiondb.ingredient \
                        WHERE lang = 'en') as a \
                        ON product.id = a.product_id;"



# MAIN APPLICATION

# Query the database and print the data

df = query_eatfit_db(nutri_score_ingr_en)

print(df.head())

# Check the existence of NaN values and the rating distribution
print("\nThe following number of cells is empty")
print(df.isnull().sum())

# Drop empty values
df.dropna(inplace=True)

# Delete other data
df = df[df.text != 'Product information’s are not available in English']

# print some information about the data

print(" \n The number of entries are: " + str(len(df.index)) + "\n")
print("Nutri-score rating distribution across the dataset:")
print(df['nutri_score_calculated'].value_counts())


# Clean the ingredient list text

cleaned_dt = data_cleaning(df)

print(cleaned_dt.head())

# ------------------MODEL-------------------------

# Split the data into train & test sets

X = df['text']
y = df['nutri_score_calculated']

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