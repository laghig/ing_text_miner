import pandas as pd
import datetime as dt

# own imports
from data_handler.Eatfit_Data_loader import *
from data_handler.OFF_data_loader import *
from data_handler.data_cleaning import *
from Model.model_comp import *
from visualization.data_summary import *
from visualization.class_plots import plot_confusion_matrix
from Model.utils import eatfit_data_summary, check_for_NaN_values


"""
Main file in which all the model steps are called in a successive order
"""

if __name__ == "__main__":

    # set the working directory
    path = r"C:\Users\Giorgio\Desktop\ETH\Code"
    os.chdir(path)

    # Define locations
    saveLoc = '/output/classification_reports/'
    ReportName = str('classification_report-' + dt.datetime.now().strftime('%Y-%m-%d-%H-%M') + '.txt')
    class_report_path = saveLoc + ReportName

    # Load the parameters file
    if os.path.exists(os.getcwd() +'\Build\model_params.yml'):
        with open(os.getcwd() +'\Build\model_params.yml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print('Parameters file is missing.')

# ------------------  DATA LOADING ------------------------
    language = params['Language']
    if params['ReloadData'] is True:

        if params['Database'] == 'Eatfit':
            column = 'text'
            # Load data from the Eatfit SQL database
            df = query_eatfit_db.query(params['Data'], language) 

        elif params['Database'] == 'OpenFoodFacts':
            # Load data from the OFF mongodb database
            DOMAIN = 'localhost'
            PORT = 27017
            column = 'ingredients_text_{}'.format(language)
            OFF_db = connect_mongo(DOMAIN, PORT)
            if language =='de':
                df = query_off_data_de(OFF_db)
            elif language == 'en':
                df = query_off_data_en(OFF_db)
            else:
                None

        print('Data retrieved')
        print(df.head())

# ---------------- DATA CLEANING -------------------------

        # Drop rows without an ingredients list
        df = clean_dataframe(df, params['Database'], language)

        # #check for empty values
        # df = check_for_NaN_values(df)

        # Print a summary of the data
        # text= eatfit_data_summary(df)
        # print(text)

        # Clean the ingredient list text
        cleaned_dt = text_cleaning(df, language, column, params['ModelModifications'])
        print('Data cleaned')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):
            print(cleaned_dt.head())

    # ---- outcomment to generate ingredients frequncy reports -----
        # ing_frequency_report(cleaned_dt['text'])

    # --------------- save interim data --------------------
        if params['Database']=='Eatfit':
            cleaned_dt.to_pickle(os.getcwd() +'/interim_results/cleaned_data.pkl')
            # cleaned_dt.to_csv(os.getcwd() +'/interim_results/cleaned_data.csv')
        else:
            cleaned_dt.to_pickle(os.getcwd() +'/interim_results/cleaned_data_OFF.pkl')

    else:
        if params['Database']=='Eatfit':
            cleaned_dt = pd.read_pickle(os.getcwd() +'/interim_results/cleaned_data.pkl')
        else:
            cleaned_dt = pd.read_pickle(os.getcwd() +'/interim_results/cleaned_data_OFF.pkl')


# -------------------- DATA SELECTION -----------------------------
    # filter data above a certain threshold
    # cleaned_dt = cleaned_dt[cleaned_dt['kg_CO2eq_pro_kg'] <= 10]

    if params['Database']=='Eatfit':
        if params['ModelParameters']['approach']== 'classification':
            X = cleaned_dt['text']
            y = cleaned_dt['co2_score'] # ,'major_category_id' co2_score / ubp_score
        elif params['ModelParameters']['approach']== 'regression':
            X = cleaned_dt['text']
            y = cleaned_dt['kg_CO2eq_pro_kg'] # kg_CO2eq_pro_kg / ubp_pro_kg
    else:
        X = cleaned_dt['ingredients_text_{}'.format(params['Language'])]
        y = cleaned_dt['ecoscore_grade']


    # ------------------------ MODEL ------------------------------

    model = ModelStructure(X,y, params['ModelParameters'], params['ModelModifications'])

    model.assemble()
    # model.visualize()
    if params['ModelParameters']['Hyperparameter_opt'] is not True:
        model.report()
        # save_predictions_to_csv(model.X_test, model.y_test, model.predictions)

        # ----------------------- REPORT -----------------------------

        txt_block = [
            str("Date: " + dt.datetime.now().strftime('%d/%m/%Y %H:%M')),
            str("Database: " + params['Database']),
            str("Language: " + params['Language']), '\n', 
        ]

        txt_block += model.txt_block

        with open(os.getcwd() + class_report_path, 'w') as f:
            for txt in txt_block:
                f.write(str(txt))
                f.write('\n')

# --------------plots---------
# Data distribution:
# plot_class_count_hist(data='OFF') # Options: 'eatfit' / 'OFF'

# Prediction error scatter plot
# reg_scatter(model.y_test, model.predictions, params['ModelParameters']['algorithm'])

# Value distribution plot
# plot_value_distribution(cleaned_dt['kg_CO2eq_pro_kg'])

# Confusion Matrix
# plot_confusion_matrix(model.y_test, model.predictions, params['ModelParameters']['algorithm'])

print("*** Completed successfully ***")