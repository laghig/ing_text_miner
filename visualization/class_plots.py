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
    'group': ['Naive Bayes','Random Forest','k-nearest neighbour','D'],
    '         Precision': [0.792, 0.828, 0.792, 4],
    'Recall': [0.723, 0.81, 0.8, 34],
    'F-1 macro          ': [0.741, 0.816, 0.792, 24],
    'Accuracy': [0.76, 0.8948, 0.802, 14]
    })

# set the save location
# saveLoc = r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots"
cwd = os.getcwd()
print(cwd)

def plot_variance_explained(num_components, variance_explained):
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.plot(range(num_components),variance_explained, color='tab:blue')
    ax.grid(True)
    plt.xlabel("Number of components", fontsize=LabelFontSize)
    plt.ylabel("Cumulative explained variance", fontsize=LabelFontSize)
    plt.tick_params(labelsize=tick_label_size)
    plt.tight_layout()
    # plt.savefig(r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots\poc_components.png")
    plt.show()

def plot_confusion_matrix(Y_test, Y_preds, model):

    filename = "\output\plots\confusion_matrix_{}.png".format(model)
    classes = ['A', 'B', 'C', 'D', 'E']
    conf_mat = confusion_matrix(Y_test, Y_preds)
    #print(conf_mat)
    fig = plt.figure(figsize=(12,10))
    plt.matshow(conf_mat, cmap=plt.cm.Blues, fignum=1)
    plt.yticks(range(5), classes, fontsize = tick_label_size)
    plt.xticks(range(5), classes, fontsize = tick_label_size)
    plt.title('True categories', fontsize=LabelFontSize)
    plt.ylabel('Predicted categories', fontsize=LabelFontSize)
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
    plt.tight_layout()
    plt.savefig(cwd + filename)
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
        ax.set_xticks(angles[:-1], categories, color='grey', size=12)
        # Draw ylabels
        ax.set_rlabel_position(45)
        ax.set_yticks([0.625,0.75, 0.875], ["0.625","0.75","0.875"], color="grey", size=8)
        ax.set_ylim([0.5, 1])

    # Titles of the subplots
    ax1.set_title('Naive Bayes')
    ax2.set_title('Random Forest')
    ax3.set_title('k-Nearest Neighbour')
    
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

    
if __name__ == "__main__":
    plot_radar_charts(model_scores)