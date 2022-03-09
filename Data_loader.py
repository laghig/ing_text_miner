import string
import MySQLdb
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


# Query all major categories
maj_categories = "SELECT * FROM nutritiondb.major_category"

# Query a specific nutritional table
prod_id = 16
nutr_table =    "SELECT  name, amount, unit_of_measure, is_mixed \
                FROM nutritiondb.nutrition_fact \
                WHERE product_id = {};".format(prod_id)

# Query the ingredients list for a specific product
prod_id = 16
ingr_list=  "SELECT lang, text \
            FROM nutritiondb.ingredient \
            WHERE product_id = {};".format(prod_id)
# in this case would not be better to generate a dictionary, with the language a key and the ingridients list as a value?


df = query_eatfit_db(ingr_list)
print(df)