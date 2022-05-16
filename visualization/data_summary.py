import pandas as pd
import os
import matplotlib.pyplot as plt

# Layout variables
LabelFontSize = 14
TitleFontSize = 16

def plot_class_count_hist(data):
    path = r"C:\Users\Giorgio\Desktop\ETH\Code\interim_results\cleaned_data_OFF.pkl"
    # os.chdir(path)
    # Data location
    # eatfit_path = os.path.join(os.getcwd(),'\interim_results\cleaned_data.pkl') # 
    # OFF_path = os.path.join(os.getcwd(),'/interim_results/cleaned_data_OFF.pkl')
    # print(eatfit_path)

    # Load the cleaned dataset
    df = pd.read_pickle(path)

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
    # plt.savefig(r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots\class_distribution_OFF.png") # uncomment to save the plot 
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


if __name__ == "__main__":

    print(os.getcwd())

    plot_class_count_hist(data='OFF')