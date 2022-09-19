# Environmental score predictor for food products

This application uses natural language processing and machine learning to predict the environmental footprint of food products based on packaging information. The environmental impact can be estimated with both regression and classification models. The main purpose of the project was to automate the classification of food products in a front-of-package label scheme.

## Setup
### Data
- Eatfit database from the Auto-ID lab at ETH - available upon request
- Open Food Facts (OFF) Database: https://ch-en.openfoodfacts.org/

Database connection:
- Download the MongoDB dump of the OFF database from https://world.openfoodfacts.org/data
- The OFF database was queried using the MongoDB Client of the pymongo library
- The Eatfit database was queried as a MySQLConnection object on the local server

### Installation
- Create a new virtual environment and install the required libraries
```
python -m venv venvname
source bin/activate # Linux command
venv\Scripts\activate # Windows command
pip install -r requirements.txt
```
- Authentication credentials must be added in a separate config.yml file in the Build directory.
- NLTK require an additional file to remove stop words from text. This file is available here https://www.nltk.org/nltk_data/ as Stopwords Corpus. Otherwise, the Stopword corpus can also be directly downloaded using the NLTK corpus downloader:
```
nltk.download('stopwords')
``` 
Just outcomment the command in the script data_cleaning.py. 

## Project structure

**Main scripts**
- main.py: main script where all model steps are called in a successive order
- config.yml: config yaml file, which has all the authentication credentials
- model_params.yml: yaml file where the data and model to execute can be specified

**File tree**
```
.
├── Build
    ├── requirements.txt
    ├── model_params.yml
├── model
  ├── model_comp.py           # Model Object
  ├── utils.py                # Helper functions
├── data_handler     
  ├── Eatfit_Data_loader.py   # MySQLConnection to the local server
  ├── OFF_data_loader.py      # MongoDB client
  ├── bags_of_words.py        # List of stopwords specific for the food domain
  ├── data_cleaning.py        # Data cleaning functions
├── interim_results           # contains interim cleaned data in plk format. This data is newly generated if the parameter ReloadData is set to True
├── output                    # All outputs files of the experiments 
  ├── plots                   # graphics
  ├── classification_reports  # text files with the performance metrics of each single run
├── visualization
  ├── class_plots.py          # visualizations of the classification performance
  ├── data_summary.py         # Exploratory data analysis
  ├── reg_plots.py            # visualizations of the regression performance
  ├── roc_curve.py            # Receiver-Operating-Characteristics curve for multi-class classifiers

```

## Execution
1. Specify the model pipeline in the model_params.yml file
2. Run the main.py script


