from re import I
from math import pi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# Layout variables
LabelFontSize = 22
TitleFontSize = 18
tick_label_size = 18

model_scores = pd.DataFrame({
    'group': ['Naive Bayes','Random Forest','k-nearest neighbour'],
    '         Precision': [0.774, 0.828, 0.782],
    'Recall': [0.779, 0.84, 0.759],
    'F-1 macro          ': [0.774, 0.831, 0.738],
    'Accuracy': [0.778, 0.839, 0.74]
    })

# set the save location
# saveLoc = r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots"
cwd = os.getcwd()
# print(cwd)

def plot_variance_explained(num_components, variance_explained, model):
    file_name = "\output\plots\confusion_matrix_{}_{}.png".format(model, str(num_components))
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.plot(range(num_components),variance_explained, color='tab:blue')
    ax.grid(True)
    plt.xlabel("Number of components", fontsize=LabelFontSize)
    plt.ylabel("Cumulative explained variance", fontsize=LabelFontSize)
    plt.tick_params(labelsize=tick_label_size)
    plt.tight_layout()
    plt.savefig(cwd + file_name)
    plt.show()

def plot_confusion_matrix(Y_test, Y_preds, model):

    filename = "\output\plots\confusion_matrix_{}_ubp_score.png".format(model)
    classes = ['A', 'B', 'C', 'D', 'E']
    conf_mat = confusion_matrix(Y_test, Y_preds)
    #print(conf_mat)
    fig = plt.figure(figsize=(12,10))
    plt.matshow(conf_mat, cmap=plt.cm.Blues, fignum=1)
    plt.yticks(range(5), classes, fontsize = tick_label_size)
    plt.xticks(range(5), classes, fontsize = tick_label_size)
    plt.title('Predicted class', fontsize=LabelFontSize)
    plt.ylabel('True class', fontsize=LabelFontSize)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=tick_label_size) ;
    
    for i in range(5):
        for j in range(5):
            if conf_mat[j,i]<10:
                pos = i - 0.06
            elif conf_mat[j,i]<100:
                pos = i - 0.15
            else:
                pos = i -0.22
            plt.text(pos,j+0.1, str(conf_mat[j, i]), color='black', fontsize = LabelFontSize)
    # plt.savefig(cwd + filename)
    plt.show()

def plot_radar_charts(model_scores):

    filename = "\output\plots\class_radar_chart.png"

    # number of variable
    categories=list(model_scores)[1:]
    N = len(categories)
    
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values1=model_scores.loc[0].drop('group').values.flatten().tolist()
    values1 += values1[:1]
    # values
    values2=model_scores.loc[1].drop('group').values.flatten().tolist()
    values2 += values2[:1]
    values3=model_scores.loc[2].drop('group').values.flatten().tolist()
    values3 += values3[:1]
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the spider plot
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,4),subplot_kw=dict(polar=True))
    axs = [ax1, ax2, ax3]
    
    # Draw one axe per variable + add labels
    for ax in axs:
        ax.set_xticks(angles[:-1], categories, size=14)
        # Draw ylabels
        ax.set_rlabel_position(45)
        ax.set_yticks([0.625,0.75, 0.875], ["0.625","0.75","0.875"], color="grey", size=10)
        ax.set_ylim([0.5, 1])

    # Titles of the subplots
    ax1.set_title('Naive Bayes', size=18)
    ax2.set_title('Random Forest', size=18)
    ax3.set_title('k-Nearest Neighbour', size=18)
    
    # Plot data
    ax1.plot(angles, values1, linewidth=1, linestyle='solid')
    ax2.plot(angles, values2, linewidth=1, linestyle='solid')
    ax3.plot(angles, values3, linewidth=1, linestyle='solid')
    
    # Fill area
    ax1.fill(angles, values1, 'b', alpha=0.1)
    ax2.fill(angles, values2, 'b', alpha=0.1)
    ax3.fill(angles, values3, 'b', alpha=0.1)

    # Show the graph
    plt.tight_layout()
    plt.savefig(cwd + filename)
    plt.show()

def category_performance_barplot(accuracy, category):
    
    # category_list = list(pd.unique(self.y['major_category_id']))
    # df = pd.DataFrame(columns=['y_test', 'prediction', 'major_category_id'])
    # df['y_test']= self.y_test['co2_score'].tolist()
    # df['prediction']= self.predictions
    # df['major_category_id']= self.y_test['major_category_id'].tolist()
    # print(df.head())

    # for category in category_list:
    #     df1 = df[df['major_category_id'] == category]
    #     print(str(category)+ ": " + str(metrics.accuracy_score(df1['y_test'],df1['prediction'])))

    x_pos = np.arange(len(category))
    filename = "\output\plots\class_performance_RF.png"

    plt.figure(figsize=(14,7))
    plt.bar(x_pos, accuracy,align='center')
    plt.xticks(x_pos, category, horizontalalignment = 'center', fontsize = tick_label_size)
    plt.yticks(fontsize = tick_label_size)
    plt.title('Category performance', weight='bold', fontsize=TitleFontSize)
    plt.ylabel('Accuracy', fontsize=LabelFontSize)
    plt.xlabel('Major category id', fontsize=LabelFontSize)
    plt.tick_params(labelsize=tick_label_size)
    plt.tight_layout()
    plt.savefig(cwd + filename)
    plt.show()

    
if __name__ == "__main__":

    plot_radar_charts(model_scores)
    
    category = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,18,19]
    accuracy = [0.9333333333333333, 1.0, 0.8484848484848485, 1.0, 0.8928571428571429, 0.8780487804878049, 0.9090909090909091, 0.8813559322033898, 0.8571428571428571, 0.7922077922077922, 0.627906976744186,
                0.8032786885245902, 0.8490566037735849, 0.9074074074074074, 0.7714285714285715, 0.8260869565217391, 0.7466666666666667]

    # category_performance_barplot(accuracy, category)