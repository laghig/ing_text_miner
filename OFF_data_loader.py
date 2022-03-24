import pandas as pd
import os



# ------------SDK option ----------------------
# import openfoodfacts
# # test the Python package - currently not working
# login_session_object = openfoodfacts.utils.login_into_OFF()

# traces = openfoodfacts.facets.get_additives()

# print(traces)


# -------------- CSV option --------------------
# # Read the csv export

# #set the working directory
# path = r"C:\Users\Giorgio\Desktop\ETH\Code"
# os.chdir(path)

# # Load the data
# df = pd.read_csv("openfoodfacts_export.csv", sep ='\t')
# data = df.loc[:, ['code', 'product_name_en', 'product_name_fr', 'ingredients_text_fr', 'ingredients_text_en', 'off:ecoscore_grade', 'off:ecoscore_score']]

# print(data.head())
# print(len(data))

# # Check the existence of NaN values and the rating distribution
# print("\nThe following number of cells is empty")
# print(data.isnull().sum())

# # Drop empty values
# print("\nDropping rows with no rating or ingredient list")
# data.dropna(subset=['ingredients_text_fr','off:ecoscore_score'], inplace=True)

# print("\nthe new datasets has " + str(len(data)) + " entries.")
# print(data.isnull().sum())

# -----------Mongo db option---------------------
from pymongo import MongoClient, errors

DOMAIN = 'localhost' # Note: this should be replaced by the URL of your own container!! 
PORT = 27017

def connect_mongo(host, port):
    # use a try-except indentation to catch MongoClient() errors
    try:
        # try to instantiate a client instance
        client = MongoClient(DOMAIN, PORT)

        OFF_db = client.off
        
    except errors.ServerSelectionTimeoutError as err:
        # set the client to 'None' if exception
        client = None

        # catch pymongo.errors.ServerSelectionTimeoutError
        print ("pymongo ERROR:", err)

    return OFF_db

OFF_db = connect_mongo(DOMAIN, PORT)

# Count documents

# print(OFF_db.products.count_documents({})) # around 2.2 Mio. documents (2'235'098)
#print(OFF_db.products.count_documents({'countries': 'Switzerland'})) # return 7967 entries
#print(OFF_db.products.count_documents({"countries": {'$regex': 'Switzerland'}})) # return 16107 entries
# print(OFF_db.products.count_documents({ '$or': [ 
#     {"countries": {'$regex': 'Switzerland'}}, {'contries':{'$regex': 'Suisse'}},
#     {"countries": {'$regex': 'Schweiz'}}, {'contries':{'$regex': 'Zwitserland'}},
#     {"countries": {'$regex': 'Svizzera'}}
#     ]})) # returns 16248 entries
#print(OFF_db.products.count_documents({"countries_tags": {'$regex': 'switzerland'}})) # returns 47407 entries

# Queries

proj = {
    'product_name_fr': True,
    'product_name_de': True,
    'product_name_en': True,
    'categories': True,
    'ingredients_text_de': True,
    'Ingredients_text_fr': True,
    'Ingredients_text_en': True,
    'countries': True,
    'countries_tags': True,
    'ecoscore_grade': True,
}

# Query to find a single document matching the query - if empty returns the first document
print(OFF_db.products.find_one({'countries': 'Switzerland'}, proj))

# Query for more than one document

# for product in OFF_db.products.find({"countries_tags": {'$regex': 'switzerland'}}):
#     print(product)