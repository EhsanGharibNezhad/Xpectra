# Module for generating visualizations of spectral data, such as plots, heatmaps, and interactive charts.

# Import functions/Classes from other modules ====================

# from io_funs import LoadSave

# Import libraries ========================================

# ******* Standard Data Manipulation / Statistical Libraries *****
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
# import pickle as pk

from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error,  mean_absolute_error
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import chi2
import os
from matplotlib.ticker import AutoMinorLocator

import pprint

# ******* Data Visulaization Libraries ****************************
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib import rcParams

from bokeh.plotting import output_notebook

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def print_results_fun(targets, print_title=None):
    """
    Print the outputs in a pretty format using the pprint library.

    Parameters
    ----------
    targets : any
        The data to be printed.
    print_title : str
        An optional title to display before the printed data.
    """

    print('*' * 30 + '\n')

    if print_title is not None:
        print(print_title+ '\n')

    # Use pprint to print the data in a well-formatted and indented manner.
    pprint.pprint(targets, indent=4, width=30)

    print('*' * 30 + '\n')


def plot_spectra_errorbar_bokeh(trained_fit_class,
                                y_label="Signal",
                                title_label=None,
                                data_type='x_y_yerr',
                                plot_type='scatter'):
    """
    Plot the spectra with error bars using Bokeh.

    Parameters
    ----------
    y_label : str, optional
        Label for the y-axis. Default is "Signal".
    title_label : str, optional
        Title of the plot. Default is None.
    data_type : str, optional
        Type of data. Default is 'x_y_yerr'.
    plot_type : str, optional
        Type of plot. Can be 'scatter' or 'line'. Default is 'scatter'.
    """

    molecule_name = trained_fit_class.absorber_name
    x_obs = trained_fit_class.wavelength_values
    y_obs = trained_fit_class.signal_values
    y_obs_err = getattr(trained_fit_class, 'signal_errors', None)  # Assuming signal_errors is an attribute

    # Create the figure
    p = figure(title=f"{molecule_name}: Calibrated Laboratory Spectra" if title_label is None else title_label,
               x_axis_label="Wavelength [𝜇m]",
               y_axis_label=y_label,
               width=1000, height=300,
               y_axis_type="linear",
               tools="pan,wheel_zoom,box_zoom,reset")

    if plot_type == 'scatter':
        # Add the scatter plot
        p.scatter(x_obs, y_obs, size=4, fill_color='green', line_color=None, line_alpha=0.2,
                  legend_label=f"{molecule_name}: Laboratory Spectra")
    elif plot_type == 'line':
        # Add the line plot
        p.line(x_obs, y_obs, line_width=2, line_color='green', alpha=0.6,
               legend_label=f"{molecule_name}: Laboratory Spectra")

    if data_type == 'x_y_yerr' and y_obs_err is not None:
        # Define maximum error threshold as a percentertage of y-value
        max_error_threshold = 0.8

        # Calculate adjusted error bar coordinates
        upper = np.minimum(y_obs + y_obs_err, y_obs + y_obs * max_error_threshold)
        lower = np.maximum(y_obs - y_obs_err, y_obs - y_obs * max_error_threshold)

        # Add error bars to the plot
        p.segment(x0=x_obs, y0=lower, x1=x_obs, y1=upper, color='gray', line_alpha=0.7)

    # Increase size of x and y ticks
    p.title.text_font_size = '14pt'
    p.xaxis.major_label_text_font_size = '14pt'
    p.xaxis.axis_label_text_font_size = '14pt'
    p.yaxis.major_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_size = '14pt'

    # Show the plot
    output_notebook()
    show(p)


def plot_spectra_errorbar_seaborn(trained_fit_class,
                                  y_label="Signal",
                                  title_label=None,
                                  data_type='x_y_yerr',
                                  plot_type='scatter'):
    """
    Plot the spectra with error bars using Seaborn.

    Parameters
    ----------
    y_label : str, optional
        Label for the y-axis. Default is "Signal".
    title_label : str, optional
        Title of the plot. Default is None.
    data_type : str, optional
        Type of data. Default is 'x_y_yerr'.
    plot_type : str, optional
        Type of plot. Can be 'scatter' or 'line'. Default is 'scatter'.
    """

    # Set a font that includes the μ glyph
    rcParams['font.sans-serif'] = ['DejaVu Sans']
    rcParams['axes.unicode_minus'] = False

    molecule_name = trained_fit_class.absorber_name
    x_obs = trained_fit_class.wavelength_values
    y_obs = trained_fit_class.signal_values
    y_obs_err = getattr(trained_fit_class, 'signal_errors', None)  # Assuming signal_errors is an attribute

    fig, ax1 = plt.subplots(figsize=(10, 4))

    if plot_type == 'scatter':
        sns.scatterplot(x=x_obs, y=y_obs, color='green', s=40, alpha=0.6,
                        label=f"{molecule_name}: Laboratory Spectra", ax=ax1)
    elif plot_type == 'line':
        sns.lineplot(x=x_obs, y=y_obs, color='green', linewidth=2, alpha=0.6,
                     label=f"{molecule_name}: Laboratory Spectra", ax=ax1)

    if data_type == 'x_y_yerr' and y_obs_err is not None:
        # Define maximum error threshold as a percentertage of y-value
        max_error_threshold = 0.8

        # Calculate adjusted error bar coordinates
        upper = np.minimum(y_obs + y_obs_err, y_obs + y_obs * max_error_threshold)
        lower = np.maximum(y_obs - y_obs_err, y_obs - y_obs * max_error_threshold)

        ax1.errorbar(x_obs, y_obs, yerr=[y_obs - lower, upper - y_obs], fmt='none', ecolor='gray', alpha=0.7)

    ax1.set_xlabel("Wavenumber [cm$^{-1}$]", fontsize=12)
    ax1.set_ylabel(y_label, fontsize=12)
    ax1.set_title(f"{molecule_name}: Calibrated Laboratory Spectra" if title_label is None else title_label,
                  fontsize=14)
    ax1.legend()
    ax1.grid(True)

    # Add a twin x-axis with the transformation 10^4/x
    ax2 = ax1.twiny()
    x_transformed = 10 ** 4 / x_obs
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels([f'{10 ** 4 / tick:.2f}' for tick in ax1.get_xticks()])
    ax2.set_xlabel("Wavelength [μm]", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_baseline_fitting(trained_fit_class):
    """
    Plot the original spectrum and the fitted baseline.
    """
    if trained_fit_class.fitted_baseline_params is None:
        raise ValueError("Baseline parameters have not been fitted yet. Call fit_baseline() first.")

    x = trained_fit_class.wavelength_values
    y = trained_fit_class.signal_values

    amplitude, freq, phase, offset = trained_fit_class.fitted_baseline_params

    def sine_wave(x, amplitude, freq, phase, offset):
        return amplitude * np.sin(2 * np.pi * freq * x + phase) + offset

    y_baseline = sine_wave(x, amplitude, freq, phase, offset)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Original Spectrum", color="blue")
    plt.plot(x, y_baseline, label="Fitted Baseline", color="red", linestyle="--")
    plt.xlabel("Wavelength [µm]")
    plt.ylabel("Signal")
    plt.title("Spectrum with Fitted Baseline")
    plt.legend()
    plt.show()



