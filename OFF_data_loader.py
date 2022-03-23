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

# use a try-except indentation to catch MongoClient() errors
try:
    # try to instantiate a client instance
    client = MongoClient(DOMAIN, PORT)

    db = client.off
    
except errors.ServerSelectionTimeoutError as err:
    # set the client to 'None' if exception
    client = None

    # catch pymongo.errors.ServerSelectionTimeoutError
    print ("pymongo ERROR:", err)

db.products

# Count the documents 
# print(db.products.count_documents({})) # around 2.2 Mio. documents (2'235'098)
#print(db.products.count_documents({'countries': 'Switzerland'})) # return 7967 entries

#print(db.products.count_documents({"countries": {'$regex': 'Switzerland'}})) # return 16107 entries
#print(db.products.count_documents({"countries": {'$regex': "/%" + 'Switzerland' + "/"}})) # zero entries

# Queries

# Query to find a single document matching the query - if empty returns the first document
# print(db.products.find_one({'countries': 'Switzerland'}))

# Query for more than one document

# for product in db.products.find({'countries': 'Switzerland'}):
#     print(product)