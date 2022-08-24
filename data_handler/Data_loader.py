import string
import mysql.connector
from mysql.connector import Error
from numpy import float64, int64
from collections import Counter
import numpy as np
import yaml
import pandas as pd
import sys
import os

# pd.set_option('display.float_format', lambda x: '%.0f' % x)

class Eatfit_data:

    def __init__(self, sql_query):
        """
        Class to fetch data form the eatfit db

        args:
            sql_query: SQL query as a string
        """
        self.sql_query = sql_query
        self.df = pd.DataFrame()

    def connect_eatfit_db(self, sql_query=string):
        """
        try-except indentation to connect to the eatfit db, queries are returned as pandas dataframe 

        Args:
            sql_query (string): SQL query as a string 
        returns:
            pandas dataframe: queried data
        """
        #set the working directory
        path = r"C:\Users\Giorgio\Desktop\ETH\Code"
        os.chdir(path)

        # Load the configuration file
        if os.path.exists("config.yml"):
            with open(os.getcwd() +'\config.yml') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            print('Config file is missing.')

        try:
            connection = mysql.connector.connect(host=config['mysql_db']['Host'],
                                user=config['mysql_db']['User'],
                                passwd=config['mysql_db']['Password'],
                                db=config['mysql_db']['Database'])

            if connection.is_connected():
                db_Info = connection.get_server_info()
                print("Connected to MySQL Server version ", db_Info)
                cursor = connection.cursor()

                # SQL queries
                cursor.execute(self.sql_query)
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
        

class query_eatfit:
        """
        Class to query the database depending on the specified parameters
        """
        def __init__(self):
            pass

        def query(self, data, language):
            
            if data == 'ingr_ubp_score':
                df = self.__ingr_ubp_score(language)
            elif data == 'ingr_nutriscore':
                df = self.__ingr_nutri_score()
            elif data == 'lca-data':
                df = self.query_lca_data()
            elif data == 'ing_text':
                df = self.query_ing_text(language)

            return df

        def __ingr_ubp_score(self, language):
            sql_query = "SELECT b.product_id, bls_lca.bls_code, ubp_pro_kg, kg_CO2eq_pro_kg, ubp_score, co2_score, lca_description, b.text, b.lang \
                        FROM bls_lca INNER JOIN \
                        (SELECT  BLS_Code, a.product_id, prod_id, a.text, a.lang \
                        FROM  nutritiondb.lca_match_products_v3 \
                        INNER JOIN \
                        (SELECT *  \
                        FROM nutritiondb.ingredient \
                        WHERE lang ='{}') as a \
                        ON lca_match_products_v3.prod_id = a.product_id) as b \
                        ON b.BLS_Code = bls_lca.bls_code;".format(language)
            
            eatfit_db = Eatfit_data(sql_query)
            df = eatfit_db.connect_eatfit_db()
            return df
        
        def __ingr_nutri_score(self):
            sql_query = "SELECT  product.id, a.text, a.lang, product.nutri_score_calculated \
                        FROM  nutritiondb.product \
                        LEFT JOIN \
                        (SELECT * \
                        FROM nutritiondb.ingredient \
                        WHERE lang = 'en') as a \
                        ON product.id = a.product_id;"
            query_db = Eatfit_data(sql_query)
            df = query_db.connect_eatfit_db()
            return df

        def query_lca_data(self):
            sql_query = "SELECT b.gtin, b.product_name_de, a.BLS_Code FROM \
                        nutritiondb.bls_matching_prod_zahw as a \
                        INNER JOIN nutritiondb.product as b on prod_id = id;"
            query_db = Eatfit_data(sql_query)
            df = query_db.connect_eatfit_db()
            return df
        
        def query_ing_text(self, language):
            sql_query = "SELECT *  \
                        FROM nutritiondb.ingredient \
                        WHERE lang = '{}'".format(language)
            query_db = Eatfit_data(sql_query)
            df = query_db.connect_eatfit_db()
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

query_eatfit_db = query_eatfit()

if __name__ == "__main__":

    df1 =  query_eatfit_db.query(data='ingr_ubp_score', language='de')
    
    # df1 = query_eatfit_db.query_lca_data()
    # df1['gtin']=df1['gtin'].astype(int64)
    # df1['gtin']=df1['gtin'].astype("string")

    # print(df1['co2_score'].head())
    # max_gtin = df1['gtin'].max()
    # print(df1.head())
    # print(max_gtin)

    # Generate a list of words included as first ingredient
    df= df1['text']
    df= df.apply(lambda x: ",".join(x.split(",",1)[0]))
    df = df.apply(lambda x: x.lower())
    df= df.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    # results = Counter()
    first_ing = ' '.join([i for i in df]).split()

    # first_ing.apply(results.update)
    # results = results.most_common()
    # unique_df = pd.DataFrame(results, columns = ['word', 'count'])
    # myset = set(first_ing)
    unique_df = pd.value_counts(np.array(first_ing))
    df2 = unique_df.to_frame(name='Count')
    df2 = df2[df2.Count != 1]
    df2 = df2[df2.Count != 2]
    # unique_df.to_csv(r"C:\Users\Giorgio\Desktop\unique_df.csv")
    key_words = df2.index.to_list()
    print(key_words)

    


    # df2 = pd.read_csv(r"C:\Users\Giorgio\Desktop\ETH\Master Thesis\off_data.csv")
    # df2.dropna(subset=['id'],inplace=True)
    # df2['id']=df2['id'].astype(float)
    # max_gtin2 = df2['id'].max()
    # print(max_gtin2)
    # df2=df2[df2['id']<=max_gtin +1]
    # df2['id']=df2['id'].astype(float64)
    # max_gtin2 = df2['id'].max()
    # print(df1['gtin'])
    # # df2.dropna(subset=['id'],inplace=True)

    # df2['id']=df2['id'].astype(int64)

    # inner_merge = pd.merge(df1, df2, left_on='gtin', right_on='id')
    # inner_merge.to_csv(r"C:\Users\Giorgio\Desktop\ETH\Master Thesis\off_Eatfit_merge.csv")

    #print(df.head())