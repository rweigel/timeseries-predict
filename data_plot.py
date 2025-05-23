"""
Process pre-serialized data stored in .pkl files (from data.py), compute derived
quantities, and generate time series plots of positional and magnetic field vectors in both
Cartesian and polar coordinate systems.

Main functionalities include:
- Reading .pkl files containing spacecraft position and magnetic field data
- Computing vector magnitudes and angular coordinates (theta, phi) in both spatial and field vectors
- Merging timestamp columns into a datetime index
- Plotting and saving figures of both raw and derived data for each file
- Running in parallel using ProcessPoolExecutor to speed up batch processing

Generated plots are saved to a created local directory ('./data_plot') and include:
- Cartesian position (x, y, z)
- Polar position (r, θ, φ)
- Magnetic field components (Bx, By, Bz)
- Polar magnetic field (B, θ_b, φ_b)

"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from concurrent.futures import ProcessPoolExecutor
import time

def plot_data(dataframe, columns, labels, save_path, title, suptitle):
    """
    Plot selected columns from a DataFrame and save the figure as an image.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing time series data.
        columns (list of str): Column names to plot.
        labels (list of str): Y-axis labels corresponding to each column.
        save_path (str): Full path where the plot image will be saved.
        title (str): Title used in console output.
        suptitle (str): Title displayed on the plot.

    Returns:
        None
    """
    plt.ioff()  # Turn off interactive mode
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(suptitle, fontsize=16)

    for ax, col, label in zip(axs, columns, labels):
        ax.plot(dataframe['date'], dataframe[col], linestyle='-')
        ax.set_ylabel(label)
        ax.grid(True)
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    axs[-1].set_xlabel('Date')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {title} plot to {save_path}")


def process_file(file_path):
    """
    Read a .pkl data file, compute derived quantities, generate plots, and save them.

    Args:
        file_path (str): Path to the pickle file containing spacecraft data.

    Returns:
        None
    """
    dataframe = pd.read_pickle(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Processing {file_name}")

    # Compute vector magnitudes and angles
    x, y, z = dataframe['x[km]'], dataframe['y[km]'], dataframe['z[km]']
    bx, by, bz = dataframe['bx[nT]'], dataframe['by[nT]'], dataframe['bz[nT]']

    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.sqrt(x**2 + y**2 + z**2)
        b = np.sqrt(bx**2 + by**2 + bz**2)
        dataframe['r[km]'] = r
        dataframe['theta[rad]'] = np.arctan2(y, x)
        dataframe['phi[rad]'] = np.arccos(z / r)
        dataframe['b[nT]'] = b
        dataframe['theta_b[rad]'] = np.arctan2(by, bx)
        dataframe['phi_b[rad]'] = np.arccos(bz / b)

    # Combine datetime columns
    dataframe['date'] = pd.to_datetime(dataframe[['year', 'month', 'day', 'hour', 'minute', 'second']])

    # Define plot groups
    plot_groups = [
        (['x[km]', 'y[km]', 'z[km]'], ['x (km)', 'y (km)', 'z (km)'], f"{file_name}_position.png", "Position", f"{file_name} - Position"),
        (['r[km]', 'theta[rad]', 'phi[rad]'], ['r (km)', 'φ (rad)', 'θ (rad)'], f"{file_name}_position_polar.png", "Polar Position", f"{file_name} - Position, Polar"),
        (['bx[nT]', 'by[nT]', 'bz[nT]'], ['Bx (nT)', 'By (nT)', 'Bz (nT)'], f"{file_name}_bvalues.png", "Magnetic Field", f"{file_name} - Magnetic Field"),
        (['b[nT]', 'theta_b[rad]', 'phi_b[rad]'], ['B (nT)', 'φ (rad)', 'θ (rad)'], f"{file_name}_bvalues_polar.png", "Polar Magnetic Field", f"{file_name} - Magnetic Field, Polar")
    ]

    save_directory = './data_plot'
    for columns, labels, save_filename, title, suptitle in plot_groups:
        save_path = os.path.join(save_directory, save_filename)
        plot_data(dataframe, columns, labels, save_path, title, suptitle)


def main():
    """
    Main function that finds all .pkl files in the data directory and processes them in parallel.

    Returns:
        None
    """
    data_directory = './data'
    save_directory = './data_plot'
    os.makedirs(save_directory, exist_ok=True)

    file_paths = glob.glob(os.path.join(data_directory, '*.pkl'))

    # Use parallel processing
    with ProcessPoolExecutor() as executor:
        executor.map(process_file, file_paths)

    print("All plots created from directory")


if __name__ == "__main__":

    start = time.time()

    main()

    print("Block time:", time.time() - start)
