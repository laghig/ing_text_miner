import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Layout variables
LabelFontSize = 22
TitleFontSize = 18
tick_label_size = 18

cwd = os.getcwd()

def reg_scatter(y_test, predictions, model):
    filename = "/output/plots/reg_scatter_{}_ubp.png".format(model)
    fig, ax = plt.subplots(figsize=(16,7))
    ax.scatter(y_test, predictions, edgecolors=(0, 0, 0), alpha=0.5, s=50)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2,)
    ax.set_xlabel("True score", fontsize=LabelFontSize)
    ax.set_ylabel("Predicted score", fontsize=LabelFontSize)
    ax.set_title('{} predictions against true values'.format(model), weight='bold', fontsize=TitleFontSize)
    plt.tick_params(labelsize=tick_label_size)
    plt.savefig(cwd + filename)
    plt.show()

def plot_residual_plot(y_test,y_pred):
    residuals = y_test - y_pred
    plt.scatter(residuals,y_pred)
    plt.show()

def plot_reg_coeff(reg_coefficients, model):
    filename = "/output/plots/reg_coeff_{}_final.png".format(model)
    ingr = list(zip(*reg_coefficients))[0]
    coef = list(zip(*reg_coefficients))[1]
    x_pos = np.arange(len(ingr))

    plt.figure(figsize=(14,7))
    plt.bar(x_pos, coef,align='center')
    plt.xticks(x_pos, ingr, rotation=45, horizontalalignment = 'right', fontsize = tick_label_size)
    plt.yticks(fontsize = tick_label_size)
    plt.title('{} regression coefficients'.format(model), weight='bold', fontsize=TitleFontSize)
    plt.ylabel('Coefficients', fontsize=LabelFontSize)
    plt.tick_params(labelsize=tick_label_size)
    plt.tight_layout()
    plt.savefig(cwd + filename)
    plt.show()