import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Layout variables
LabelFontSize = 18
TitleFontSize = 16
tick_label_size = 14

def plot_class_count_hist(data):
    path = r"C:\Users\Giorgio\Desktop\ETH\Code\interim_results\cleaned_data.pkl"
    # os.chdir(path)
    # Data location
    # eatfit_path = os.path.join(os.getcwd(),'\interim_results\cleaned_data.pkl') # 
    # OFF_path = os.path.join(os.getcwd(),'/interim_results/cleaned_data_OFF.pkl')
    # print(eatfit_path)

    # Load the cleaned dataset
    df = pd.read_pickle(path)

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
    # plt.savefig(r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots\class_distribution_OFF.png") # uncomment to save the plot 
    plt.show()

def plot_value_distribution(x):
    plt.figure(figsize=(16,9))
    plt.hist(x, bins=120, rwidth=0.9)
    plt.xlim(0,round(x.max())+1)
    plt.xlabel('Kg CO2 eq/kg', fontsize=LabelFontSize)
    plt.ylabel('Occurrence', fontsize=LabelFontSize)
    plt.title('Value distribution', weight='bold', fontsize=TitleFontSize)
    plt.tick_params(labelsize=tick_label_size)
    plt.tight_layout()
    plt.savefig(r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots\CO2_value_distribution.png") # uncomment to save the plot 
    plt.show()

def reg_scatter(y_test, predictions):
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=4)
    ax.set_xlabel("True score [UBP/kg]", fontsize=LabelFontSize)
    ax.set_ylabel("Predicted score [UBP/kg]", fontsize=LabelFontSize)
    ax.set_title('Prediction error', weight='bold', fontsize=TitleFontSize)
    # plt.savefig(r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots\reg_scatter.png")
    plt.show()

def plot_confusion_matrix(Y_test, Y_preds):
    classes = ['A', 'B', 'C', 'D', 'E']
    conf_mat = confusion_matrix(Y_test, Y_preds)
    #print(conf_mat)
    fig = plt.figure(figsize=(6,6))
    plt.matshow(conf_mat, cmap=plt.cm.Blues, fignum=1)
    plt.yticks(range(5), classes, fontsize = tick_label_size)
    plt.xticks(range(5), classes, fontsize = tick_label_size)
    plt.xlabel('True categories', fontsize=LabelFontSize)
    plt.ylabel('Predicted categories', fontsize=LabelFontSize)
    plt.colorbar();
    for i in range(5):
        for j in range(5):
            plt.text(i-0.2,j+0.1, str(conf_mat[j, i]), color='black', fontsize = LabelFontSize)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # print(os.getcwd())

    # plot_class_count_hist(data='eatfit')

    path = r"C:\Users\Giorgio\Desktop\ETH\Code\interim_results\cleaned_data.pkl"

    # Load the cleaned dataset
    df = pd.read_pickle(path)
    x=df['kg_CO2eq_pro_kg']
    

    plot_class_count_hist('eatfit')

    # plot_value_distribution(x)



    # DESCRIPTIVE STATISTICS FOR THE CLASSIFICATION SCHEME
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