import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Layout variables
LabelFontSize = 22
TitleFontSize = 18
tick_label_size = 18

def reg_scatter(y_test, predictions):
    fig, ax = plt.subplots(figsize=(16,7))
    ax.scatter(y_test, predictions, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=3)
    ax.set_xlabel("True score", fontsize=LabelFontSize)
    ax.set_ylabel("Predicted score", fontsize=LabelFontSize)
    ax.set_title('KNN predictions against true values', weight='bold', fontsize=TitleFontSize)
    plt.tick_params(labelsize=tick_label_size)
    # plt.savefig(r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots\reg_scatter_KNN_rs11.png")
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
    # plt.savefig(r"C:\Users\Giorgio\Desktop\ETH\Code\output\plots\reg_coeff_ridge.png")
    plt.show()