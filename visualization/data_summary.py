import pandas as pd
from collections import Counter
import os
import matplotlib.pyplot as plt
import numpy as np

# Layout variables
LabelFontSize = 22
TitleFontSize = 18
tick_label_size = 18

cwd = os.getcwd()

def plot_class_count_hist(df, data):
    file_name = "\output\plots\class_distribution_{}.png".format(data)

    if data == 'eatfit':
        class_count = df.groupby('co2_score')['co2_score'].count()
    else:
        class_count = df.groupby('ecoscore_grade')['ecoscore_grade'].count() 
    print(class_count)

    class_count.plot.bar()
    plt.xticks(rotation=0)
    plt.xlabel('Environmental score', fontsize=LabelFontSize)
    plt.ylabel('Occurrence', fontsize=LabelFontSize)
    plt.title('Data distribution', weight='bold', fontsize=TitleFontSize)
    plt.tick_params(labelsize=LabelFontSize)
    plt.tight_layout()
    # plt.savefig(cwd + file_name) # uncomment to save the plot 
    plt.show()

def plot_value_distribution(x):
    file_name = "/output/plots/gwp_value_distribution.png"
    plt.figure(figsize=(16,7))
    plt.hist(x, bins=120, rwidth=0.9)
    plt.xlim(0,round(x.max())+1) 
    plt.xlabel('GWP [Kg$CO_{2}$eq./kg]', fontsize=LabelFontSize) # GWP [Kg$CO_{2}$eq./kg] / Umweltbelastungspunkte [UBP/kg]
    plt.ylabel('Occurrence', fontsize=LabelFontSize)
    plt.title('Value distribution', weight='bold', fontsize=TitleFontSize)
    plt.tick_params(labelsize=tick_label_size)
    plt.tight_layout()
    plt.savefig(cwd + file_name) # uncomment to save the plot 
    plt.show()

def categories_dist_barplot(x):
    file_name = "\output\plots\class_distribution_Eatfit.png"
    class_count = df.groupby('major_category_id')['major_category_id'].count()
    plt.figure(figsize=(16,7))
    class_count.plot.bar()
    plt.xticks(rotation=0)
    plt.xlabel('Categories', fontsize=LabelFontSize)
    plt.ylabel('Occurrence', fontsize=LabelFontSize)
    plt.title('Categories distribution', weight='bold', fontsize=TitleFontSize)
    plt.tick_params(labelsize=LabelFontSize)
    plt.tight_layout()
    plt.savefig(cwd + file_name) # uncomment to save the plot 
    plt.show()

def ing_frequency_report(text):
    """
    Generate a csv file with words counts
    """
    results = Counter()
    text.str.split().apply(results.update)
    results = results.most_common()
    unique_df = pd.DataFrame(results, columns = ['word', 'count'])
    # save the dataframe as csv file
    unique_df.to_csv(os.getcwd() +'/interim_results/unique_ing_count.csv')

def save_predictions_to_csv(X,y,predictions):
    # save the prediction set
    df = pd.DataFrame(columns=['ingredients', 'True value', 'Predictions'])
    df['ingredients']= X.tolist()
    df['True value']= y.tolist()
    df['Predictions']= predictions
    df.to_csv(cwd +'/interim_results/prediction_data.csv')

if __name__ == "__main__":


    path = cwd +'\interim_results\cleaned_data.pkl'
    
    # Load the cleaned dataset
    df = pd.read_pickle(path)

    # Plot the class distribution
    # plot_class_count_hist(df, 'eatfit')

    # Plot the continuous data distribution
    x=df['kg_CO2eq_pro_kg'] # 'ubp_pro_kg' / 'kg_CO2eq_pro_kg'
    # plot_value_distribution(x)

    # Plot the categories distribution
    # categories_dist_barplot(df)

    # DESCRIPTIVE STATISTICS FOR THE CLASSIFICATION SCHEME
    # x=df['ubp_pro_kg']
    # print(x.quantile(0.2))
    # max = int(x.max())
    # step=int(x.max()/6)
    # classes = np.arange(0,max,step)
    # print(classes)
    # print(x.mean())
    # print(x.quantile(0.1))
    # print(x.quantile(0.3))
    # print(x.quantile(0.5))
    # print(x.quantile(0.75))