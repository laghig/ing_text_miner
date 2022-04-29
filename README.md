# Environmental score predictor for food products

This application uses natural language processing and machine learning to predict the environmental footprint of food products based on packaging information.

## Dependencies
- MongoDB
- ...

## Data
- Eatfit database from the Auto-ID lab at ETH
- Open Food Facts (OFF) Database: https://ch-en.openfoodfacts.org/

## Installation
- clone this directory
- create a new virtual environment:
```
with python:
python -m venv venvname
```
- activate the virtual environment
```
Linux:
source bin/activate 

Windows:
venv\Scripts\activate
```
- install the required libraries:
```
pip install -r requirements.txt
```
In this way should be possible to use the basic functionalities of the model. Just keep the ReloadData parameter always false.
**Data cleaning**:

nltk require an additional file to remove stop words from text. This file is available here https://www.nltk.org/nltk_data/ as Stopwords Corpus.
Otherwise, the file can be directly downloaded using the NLTK corpus downloader: from the data_cleaning.py script by outcommenting the line 
```
nltk.download('stopwords')
``` 
Just outcomment line 8 in the script data_cleaning.py. Up to now data cleaning can be tested just as a single script if the connection to the databases are not implemented.

## Database connections
- Download the OFF database from https://world.openfoodfacts.org/data