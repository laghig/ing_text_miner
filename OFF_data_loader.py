import pandas as pd
import os
import openfoodfacts

# test the Python package - currently not working
#login_session_object = openfoodfacts.utils.login_into_OFF()

#traces = openfoodfacts.facets.get_additives()

#print(traces)

# Read the csv export

#set the working directory
path = r"C:\Users\Giorgio\Desktop\ETH\Code"
os.chdir(path)

# Load the data
df = pd.read_csv("openfoodfacts_export.csv", sep ='\t')
data = df.loc[:, ['code', 'product_name_en', 'product_name_fr', 'ingredients_text_fr', 'ingredients_text_en', 'off:ecoscore_grade', 'off:ecoscore_score']]

print(data.head())
print(len(data))

# Check the existence of NaN values and the rating distribution
print("\nThe following number of cells is empty")
print(data.isnull().sum())

# Drop empty values
print("\nDropping rows with no rating or ingredient list")
data.dropna(subset=['ingredients_text_fr','off:ecoscore_score'], inplace=True)

print("\nthe new datasets has " + str(len(data)) + " entries.")
print(data.isnull().sum())