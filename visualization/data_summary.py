import pandas as pd
import os
import matplotlib.pyplot as plt

# Layout variables
LabelFontSize = 14
TitleFontSize = 16

def plot_class_count_hist(data):

    # Data location
    eatfit_path = os.path.join(os.getcwd(),'\interim_results\cleaned_data.pkl') # 
    OFF_path = os.path.join(os.getcwd(),'/interim_results/cleaned_data_OFF.pkl')


    # Load the cleaned dataset
    df = pd.read_pickle(OFF_path)

    if data == 'eatfit':
        class_count = df.groupby('ubp_score')['ubp_score'].count()
    else:
        class_count = df.groupby('ecoscore_grade')['ecoscore_grade'].count() 

    class_count.plot.bar()
    plt.xticks(rotation=0)
    plt.xlabel('Environmental label', fontsize=LabelFontSize)
    plt.ylabel('Occurrence', fontsize=LabelFontSize)
    plt.title('OFF Data distribution', weight='bold', fontsize=TitleFontSize)
    plt.tick_params(labelsize=LabelFontSize)
    plt.tight_layout()
    #plt.savefig(r"C:...\class_distribution_OFF.png") # uncomment to save the plot 
    plt.show()



if __name__ == "__main__":

    print(os.getcwd())

    plot_class_count_hist(data='OFF')