import pandas as pd
import os

"""
important: mongod.exe must be executed from the cmd first to be able to connect to MongoDB server
"""

# -----------pyMongo option---------------------
from pymongo import MongoClient, errors

def connect_mongo(host, port):
    # use a try-except indentation to catch MongoClient() errors
    try:
        # try to instantiate a client instance
        client = MongoClient(host, port)

        OFF_db = client.off
        
    except errors.ServerSelectionTimeoutError as err:
        # set the client to 'None' if exception
        client = None

        # catch pymongo.errors.ServerSelectionTimeoutError
        print ("pymongo ERROR:", err)

    return OFF_db

def query_off_data_de(OFF_db):

    proj = {
    'product_name_en': True,
    'ingredients_text_de': True,
    'id': True,
    'countries_tags': True,
    'ecoscore_grade': True,
    }

    cursor = OFF_db.products.find(
        { '$and':[ 
            {'ingredients_text_de': {'$exists': True}},
            {'$or': [
                {"countries_tags": {'$regex': 'switzerland'}},
                {'countries_tags':{'$regex': 'suisse'}}
            ]} 
        ]}
        , proj)
    list_cur = list(cursor)
    df = pd.DataFrame(list_cur)
    return df

def query_off_data_en(OFF_db):

    proj = {
    'product_name_en': True,
    'ingredients_text_en': True,
    'id': True,
    'countries_tags': True,
    'ecoscore_grade': True,
    }

    cursor = OFF_db.products.find(
        { '$and':[ 
            {'ingredients_text_en': {'$exists': True}},
            {'$or': [
                {"countries_tags": {'$regex': 'switzerland'}},
                {'countries_tags':{'$regex': 'suisse'}}
            ]} 
        ]}
        , proj)
    list_cur = list(cursor)
    df = pd.DataFrame(list_cur)
    return df

def query_off_data(OFF_db, language):

    proj = {
    'product_name_{}'.format(language): True,
    'ingredients_text_{}'.format(language): True,
    'id': True,
    'countries_tags': True,
    }

    cursor = OFF_db.products.find(
        { '$and':[ 
            {'ingredients_text_{}'.format(language): {'$exists': True}},
            {'$or': [
                {"countries_tags": {'$regex': 'switzerland'}},
                {'countries_tags':{'$regex': 'suisse'}}
            ]} 
        ]}
        , proj)
    list_cur = list(cursor)
    df = pd.DataFrame(list_cur)
    return df

def print_data_fields(OFF_db):
    cursor = OFF_db.products.find_one({})
    for document in cursor: 
        print(document.keys())


if __name__ == "__main__":
    DOMAIN = 'localhost' # Note: this should be replaced by the URL of your own container!! 
    PORT = 27017
    OFF_db = connect_mongo(DOMAIN, PORT)
    language = 'fr' # 'de'. 'en'


    df = query_off_data(OFF_db, language)
    df.to_csv(r"C:\Users\Giorgio\Desktop\ETH\Master Thesis\off_data.csv")
    # print(df.head())

    # Count documents

    # print(OFF_db.products.count_documents({})) # around 2.2 Mio. documents (2'235'098)
    #print(OFF_db.products.count_documents({'countries': 'Switzerland'})) # return 7967 entries
    #print(OFF_db.products.count_documents({"countries": {'$regex': 'Switzerland'}})) # return 16107 entries
    # print(OFF_db.products.count_documents({ '$or': [ 
    #     {"countries": {'$regex': 'Switzerland'}}, {'countries':{'$regex': 'Suisse'}},
    #     {"countries": {'$regex': 'Schweiz'}}, {'countries':{'$regex': 'Zwitserland'}},
    #     {"countries": {'$regex': 'Svizzera'}}
    #     ]})) # returns 16248 entries
    # print(OFF_db.products.count_documents({ '$or': [
    #     {"countries_tags": {'$regex': 'switzerland'}},
    #     {'countries_tags':{'$regex': 'suisse'}},
    # ]})) # returns 47407 entries

    # Queries
    # print(OFF_db.products.find_one(
    #     { '$and':[ 
    #         {'ingredients_text': {'$exists': True}},
    #         {'$or': [
    #             {"countries_tags": {'$regex': 'switzerland'}},
    #             {'countries_tags':{'$regex': 'suisse'}}
    #         ]} 
    #     ]}
    #     ))

    # Query for more than one document

    # for product in OFF_db.products.find({"countries_tags": {'$regex': 'switzerland'}}):
    #     print(product)