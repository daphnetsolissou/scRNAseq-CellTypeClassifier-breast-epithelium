import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns



def plot_metric_boxplots(results_dict):
    sns.set(style='ticks', font_scale=1.2)

    g = sns.catplot(x='Metric', y='Value', hue='Classifier', kind='box', data=results_dict, height=8, aspect=3, legend_out=False)
    g.set_axis_labels('Evaluation Metrics (Outer Cross Validation Loop)', 'Metric Value')
    g.fig.suptitle('Evaluation metrics over 50 nCV outer loop folds for all classifiers', y=1.03)
    g._legend.set_title('Classifier')

    g.ax.xaxis.labelpad = 20

    # Create the directory if it doesn't exist
    directory = 'ncv_results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the plot with unique file names
    file_name = 'metric_boxplots'
    save_path_png = os.path.join(directory, file_name + '.png')
    save_path_pdf = os.path.join(directory, file_name + '.pdf')
    
    # Check if the file already exists
    counter = 1
    while os.path.exists(save_path_png) or os.path.exists(save_path_pdf):
        file_name += '_' + str(counter)
        save_path_png = os.path.join(directory, file_name + '.png')
        save_path_pdf = os.path.join(directory, file_name + '.pdf')
        counter += 1

    # Save the plot
    g.savefig(save_path_png, dpi=300, bbox_inches='tight')
    g.savefig(save_path_pdf, dpi=300)


def plot_mccs(mcc_dict, std_errors):
    fig, ax = plt.subplots(figsize=(8, 7))
    x_labels = [key.upper() for key in mcc_dict.keys()]
    y_values = mcc_dict.values()
    error_values = std_errors.values()
    my_cmap = plt.get_cmap("tab10")
    
    # Plot the bar chart
    bar = ax.bar(x_labels, y_values, color=my_cmap.colors)
    
    # Add error bars with custom appearance
    for rect, error_value in zip(bar, error_values):
        height = rect.get_height()
        errorbar_kwargs = {
            'x': rect.get_x() + rect.get_width() / 2,
            'y': height,
            'yerr': error_value,
            'color': 'black',
            'capsize': 5,  # Adjust the cap size of the error bars
            'linewidth': 1.5,  # Adjust the line width of the error bars
            'alpha': 0.8  # Adjust the transparency of the error bars
        }
        ax.errorbar(**errorbar_kwargs)
    
    plt.title('MCC mean values comparison between different classifiers', fontsize=22)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.bar_label(ax.containers[0])
    plt.xlabel('Classifiers', fontsize=10)

    # Create the directory if it doesn't exist
    directory = 'ncv_results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the plot with unique file names
    file_name = 'mcc_compare'
    save_path_png = os.path.join(directory, file_name + '.png')
    save_path_pdf = os.path.join(directory, file_name + '.pdf')
    
    # Check if the file already exists
    counter = 1
    while os.path.exists(save_path_png) or os.path.exists(save_path_pdf):
        file_name += '_' + str(counter)
        save_path_png = os.path.join(directory, file_name + '.png')
        save_path_pdf = os.path.join(directory, file_name + '.pdf')
        counter += 1

    # Save the plot
    plt.savefig(save_path_pdf, dpi=300)
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.show()



def plot_mean_std_errors(std_dict):
    fig, ax = plt.subplots(figsize=(8, 7))
    x_labels = [key.upper() for key in std_dict.keys()]
    y_values = std_dict.values()
    my_cmap = plt.get_cmap("tab10")
    plt.bar(x_labels, y_values, color=my_cmap.colors)
    plt.title('MCC standard error comparison between different classifiers', fontsize=22)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.bar_label(ax.containers[0])
    plt.xlabel('Classifiers', fontsize=10)

    # Create the directory if it doesn't exist
    directory = 'ncv_results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the plot with unique file names
    file_name = 'std_e_compare'
    save_path_png = os.path.join(directory, file_name + '.png')
    save_path_pdf = os.path.join(directory, file_name + '.pdf')
    
    # Check if the file already exists
    counter = 1
    while os.path.exists(save_path_png) or os.path.exists(save_path_pdf):
        file_name += '_' + str(counter)
        save_path_png = os.path.join(directory, file_name + '.png')
        save_path_pdf = os.path.join(directory, file_name + '.pdf')
        counter += 1

    # Save the plot
    plt.savefig(save_path_pdf, dpi=300)
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.show()


def compare_mean_mccs(results_dict):
    """
    For each classifier calculate mean MCC score and standard error
    :param results_dict: A dictionary with keys the names of the classifiers and values dataframes with the scores of
    the ncv experiments done for each one
    :return:
    Dictionary for each classifier's mean MCC score
    """
    mccs = {}
    std_errors = {}
    for key, df in results_dict.items():
        mean_mcc = df['MCC'].mean()
        std_error_mcc = df['MCC'].std(ddof=1) / np.sqrt(df['MCC'].shape[0])
        mccs[key] = mean_mcc
        std_errors[key] = std_error_mcc
    plot_mccs(mccs, std_errors)
    plot_mean_std_errors(std_errors)
    return mccs