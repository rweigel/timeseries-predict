"""
Model Evaluation Plotting Script

This script provides utilities for visualizing and saving the results of model 
predictions compared to actual data, specifically for time series outputs from 
bootstrap runs. It includes functions to generate comparison plots of actual 
vs. predicted time series and optionally plot ARV values across training epochs.

"""
import os
import matplotlib.pyplot as plt

def savefig(plt, save_path):
    """
    Save a matplotlib figure to a specified path, creating directories if needed.

    Parameters:
    ----------
    plt : module
        The matplotlib.pyplot module containing the current figure to be saved.

    save_path : str
        The file path where the figure will be saved. Directories are created if 
        they do not exist.

    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    plt.savefig(f"{save_path}")
    print(f"      Wrote:   {save_path}")
    plt.close()


def plot(boots, save_path, stats):
    """
    Plot actual vs. predicted time series for each model and save the figures.

    For a selected bootstrap run (currently boot_num = 0), this function plots the actual
    and predicted values for each model across each output variable, saving the resulting
    plots to disk. If epoch statistics are present, it also plots and saves the ARV over epochs.

    Parameters:
    ----------
    boots : list of dicts
        A list where each element is a dictionary containing actual and model-predicted data 
        for a single bootstrap run.

    save_path : str
        The base path where plot images will be saved. Filenames are extended with model and bootstrap identifiers.

    stats : any
        Placeholder for additional statistics, currently unused in the function.
    """
    #print(stats)
    boot_num = 0
    actual = boots[boot_num]['actual']
    outputs = actual.columns[1:]  # Exclude timestamp

    models = list(boots[0].keys())
    models.remove('actual')
    for model in models:
        predicted = boots[boot_num][model]['predicted']
        plt.figure(figsize=(8.5, 12), facecolor='white')
        gs = plt.gcf().add_gridspec(len(outputs))
        axs = gs.subplots(sharex=True)
        for i, output in enumerate(outputs):
            axs[i].plot(actual['timestamp'], actual[output], linestyle='-', color='black', label='Actual')
            axs[i].plot(predicted['timestamp'], predicted[output], linestyle='-', color='red', label=model)
            if i == 0:
                axs[i].legend()
            axs[i].set_ylabel(output)
            axs[i].grid(True)
            #datetick()

        plt.tight_layout()
        savefig(plt, f"{save_path}_boot_{boot_num}_{model}.png")

        if 'epochs' in boots[boot_num][model]:
            plt.figure(figsize=(8.5, 8.5), facecolor='white')
            plt.semilogy(boots[boot_num][model]['epochs'])
            plt.grid(True)
            plt.ylabel('ARV')
            plt.xlabel('Epoch')
            plt.tight_layout()
            savefig(plt, f"{save_path}_boot_{boot_num}_{model}_epochs.png")
