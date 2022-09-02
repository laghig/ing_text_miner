import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Layout variables
LabelFontSize = 22
TitleFontSize = 18
tick_label_size = 18

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
    plt.figure(figsize=(16,7))
    plt.hist(x, bins=120, rwidth=0.9)
    plt.xlim(0,round(x.max())+1)
    plt.xlabel('GWP [Kg$CO_{2}$eq./kg]', fontsize=LabelFontSize)
    plt.ylabel('Occurrence', fontsize=LabelFontSize)
    plt.title('Value distribution', weight='bold', fontsize=TitleFontSize)
    plt.tick_params(labelsize=tick_label_size)
    plt.tight_layout()
    plt.savefig(r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots\CO2_value_distribution.png") # uncomment to save the plot 
    plt.show()

def reg_scatter(y_test, predictions):
    fig, ax = plt.subplots(figsize=(16,7))
    ax.scatter(y_test, predictions, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=3)
    ax.set_xlabel("True score", fontsize=LabelFontSize)
    ax.set_ylabel("Predicted score", fontsize=LabelFontSize)
    ax.set_title('KNN predictions against true values', weight='bold', fontsize=TitleFontSize)
    plt.tick_params(labelsize=tick_label_size)
    plt.savefig(r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots\reg_scatter_KNN_rs11.png")
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

def plot_residual_plot(y_test,y_pred):
    residuals = y_test - y_pred
    plt.scatter(residuals,y_pred)
    plt.show()

def plot_reg_coeff(reg_coefficients):
    ingr = list(zip(*reg_coefficients))[0]
    coef = list(zip(*reg_coefficients))[1]
    x_pos = np.arange(len(ingr))

    plt.figure(figsize=(14,7))
    plt.bar(x_pos, coef,align='center')
    plt.xticks(x_pos, ingr, rotation=90, fontsize = tick_label_size)
    plt.yticks(fontsize = tick_label_size)
    plt.title('Ridge regression coefficients', weight='bold', fontsize=TitleFontSize)
    plt.ylabel('Coefficients', fontsize=LabelFontSize)
    plt.tick_params(labelsize=tick_label_size)
    plt.tight_layout()
    plt.savefig(r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots\reg_coeff_ridge.png")
    plt.show()


if __name__ == "__main__":

    # print(os.getcwd())

    # plot_class_count_hist(data='eatfit')

    path = r"C:\Users\Giorgio\Desktop\ETH\Code\interim_results\cleaned_data.pkl"

    # Load the cleaned dataset
    df = pd.read_pickle(path)
    x=df['kg_CO2eq_pro_kg']
    
    # Uncomment to plot the class distribution
    # plot_class_count_hist('eatfit')

    # Uncomment to plot the continuous data distribution
    # plot_value_distribution(x)

    # Uncomment for the coefficients plot
    reg_coefficients = [('rindfleisch', 29.205953500796138), ('kalbfleisch', 21.724724953393142), ('lammfleisch', 18.002839246624), ('kalbsfleisch', 14.115437561380901), ('kakaobutt', 13.993480416501363), ('rindfleischextrak', 13.540567660887545), ('kakao', 11.023142890445785), ('reifenkultur', 10.313785579269071), ('kaffeeboh', 10.036614902725267), ('vollmilchschokolad', 9.87934226149519), ('rostkaffee', 9.44346608662392), ('trutenfleisch', 9.319586452094807), ('butt', 8.504693622642609), ('kaliumchlorid', 8.456858796788367), ('vollmilchpulv', 8.436025082704242), ('cocoa', 8.295829390582819), ('sheafett', 8.151821863481322), ('schweinkotelett', 7.974554605657326), ('cashew', 7.954813370018415), ('ea', 7.733919735158475), ('rapslecithi', 7.6257133293676125), ('aromaextrak', 7.620793641858054), ('merlo', 7.521217747834681), ('mahl', 7.334409458168646), ('eiklar', 7.323297121226798), ('ee', 7.2606007503744445), ('pouletfleisch', 7.242851891979269), ('kondensier', 7.209859352012083), ('berggebie', 7.205416876499798), ('butterfraktio', 7.159454083094616), ('enthaltenkakao', 7.117911027406471), ('lab', 7.096613831635765), ('orangengranula', 6.973659816354033), ('waffel', 6.830104887297402), ('eisenpyrophospha', 6.683024115950112), ('schweinegra', 6.670309929910845), ('pouletschnitzel', 6.635331791054994),]
    plot_reg_coeff(reg_coefficients)



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