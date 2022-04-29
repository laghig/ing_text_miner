import string
import mysql.connector
from mysql.connector import Error
import yaml
import pandas as pd
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

# ***QUERIES***

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

bls_score_ingr_de = "SELECT  BLS_Code, a.product_id, prod_id, a.text, a.lang \
                FROM  nutritiondb.lca_data_zahw \
                INNER JOIN \
                (SELECT *  \
                FROM nutritiondb.ingredient \
                WHERE lang = 'de') as a \
                ON lca_data_zahw.prod_id = a.product_id;"


def connect_eatfit_db(sql_query=string):
    """
    try-except indentation to connect to the eatfit db, queries are returned as pandas dataframe 

    Args:
        sql_query (string): SQL query as a string 
    returns:
        pandas dataframe: queried data
    """
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

def query_eatfit_db_ingr_ubp(language):
    """
    Query to fetch a table with the upb score and the ingredients list depending on the specified language
    Input: language: string
    Output: df: pandas dataframe
    """
    query = "SELECT b.product_id, bls_lca.bls_code, ubp_score, lca_description, b.text, b.lang \
            FROM bls_lca INNER JOIN \
            (SELECT  BLS_Code, a.product_id, prod_id, a.text, a.lang \
            FROM  nutritiondb.bls_matching_prod_zahw \
            INNER JOIN \
            (SELECT *  \
            FROM nutritiondb.ingredient \
            WHERE lang ='{}') as a \
            ON bls_matching_prod_zahw.prod_id = a.product_id) as b \
            ON b.BLS_Code = bls_lca.bls_code;".format(language)
    
    df = connect_eatfit_db(query)
    return df

def query_eatfit_db_ingr_eng(query=string):
    """
    Match the SQL text, query the database, and return a pandas df
    Input: query: variable name of the SQL text
    Output: df: pandas dataframe
    """
    df = connect_eatfit_db(query)
    return df

def check_for_NaN_values(df):
    print("\n The following number of cells are empty")
    print(df.isnull().sum())
    return df

def eatfit_data_summary(df):
    # print some information about the data
    text =  " \n The number of entries are: " + str(len(df)) + "\n" 
            # "Nutri-score rating distribution across the dataset:" + str(df['nutri_score_calculated'].value_counts())
    return text