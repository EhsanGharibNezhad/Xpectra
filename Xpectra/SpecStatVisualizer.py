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
from bokeh.layouts import column

import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# UPDATE DOCS!

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


def print_fitted_parameters(fitted_params, covariance_matrices):
    """
    Print the fitted parameters and their errors for each peak as a dataframe.

    Parameters
    ----------
    fitted_params : np.ndarray
        The fitted parameters for each peak.
    covariance_matrices : np.ndarray
        The covariance matrices of the fitted parameters.
    """

    # Calculate 1-sigma error for each parameter
    peaks_info = {}
    for i, params in enumerate(fitted_params):
        peak_number = i + 1
        covariance_matrix = covariance_matrices[i]
        errors = np.sqrt(np.diag(covariance_matrix))
        rounded_errors = [round(error, 3) for error in errors]  # Round errors to 3 significant figures

        peak_info = dict(zip(['center', 'Intensity', 'width'], params))
        peak_info_with_errors = {}

        # Include ± 1-sigma error bars in the peak information
        for param_name, param_value, error in zip(['center', 'amplitude', 'width'], params, rounded_errors):
            peak_info_with_errors[param_name] = {'value': round(param_value, 3), 'error': error}

        peaks_info[f'Peak {peak_number}'] = peak_info_with_errors

    # Print peaks information
    for peak, info in peaks_info.items():
        print(peak + ':')
        for param, values in info.items():
            print(f"    {param}: {values['value']} ± {values['error']}")


def print_fitted_parameters_df(fitted_params, covariance_matrices):
    """
    Print the fitted parameters and their errors for each peak.

    Parameters
    ----------
    fitted_params : np.ndarray
        The fitted parameters for each peak.
    covariance_matrices : np.ndarray
        The covariance matrices of the fitted parameters.
    """

    # Calculate 1-sigma error for each parameter
    peaks_info = []
    for i, params in enumerate(fitted_params):
        peak_number = i + 1
        covariance_matrix = covariance_matrices[i]
        errors = np.sqrt(np.diag(covariance_matrix))
        rounded_errors = [round(error, 4) for error in errors]

        peak_info = {
            'Peak Number': peak_number,
            'center': round(params[0], 4),
            'center Error': rounded_errors[0],
            'Intensity': round(params[1], 3),
            'Intensity Error': rounded_errors[1],
            'Width': round(params[2], 3),
            'Width Error': rounded_errors[2]
        }
        peaks_info.append(peak_info)

    df = pd.DataFrame(peaks_info)
    display(df)


def plot_spectra_errorbar_bokeh(wavelength_values,
                                signal_values,
                                signal_values_err = None,
                                absorber_name = None,
                                y_label="Signal",
                                title_label=None,
                                data_type='x_y_yerr',
                                plot_type='scatter'):
    """
    Plot the spectra with error bars using Bokeh.

    Parameters
    ----------
    wavelength_values : nd.array
        Wavelength array in microns.
    signal_values : nd.array
        Signal arrays (input data).
    signal_values_err : nd.array, optional
        Error on input data.
    absorber_name : str, optional
        Molecule or atom name.
    y_label : str, optional
        Label for the y-axis. Default is "Signal".
    title_label : str, optional
        Title of the plot. Default is None.
    data_type : str, optional
        Type of data. Default is 'x_y_yerr'.
    plot_type : str, optional
        Type of plot. Can be 'scatter' or 'line'. Default is 'scatter'.
    """

    molecule_name = absorber_name
    x_obs = wavelength_values
    y_obs = signal_values
    y_obs_err = signal_values_err

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


def plot_spectra_errorbar_seaborn(wavelength_values,
                                  signal_values,
                                  signal_values_err = None,
                                  absorber_name = None,
                                  y_label="Signal",
                                  title_label=None,
                                  data_type='x_y_yerr',
                                  plot_type='scatter'):
    """
    Plot the spectra with error bars using Seaborn.

    Parameters
    ----------
    wavelength_values : nd.array
        Wavelength array in microns.
    signal_values : nd.array
        Signal arrays (input data).
    signal_values_err : nd.array, optional
        Error on input data.
    absorber_name : str, optional
        Molecule or atom name.
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

    molecule_name = absorber_name
    x_obs = wavelength_values
    y_obs = signal_values
    y_obs_err = signal_values_err

    fig, ax1 = plt.subplots(figsize=(10, 4),dpi=700)

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


def plot_baseline_fitting_seaborn(wavelength_values, 
                          signal_values, 
                          baseline_type, 
                          fitted_baseline_params, 
                          baseline_degree=None):
    """
    Plot the original spectrum and the fitted baseline using Seaborn.

    Parameters
    ----------
    wavelength_values : np.ndarray
        Wavelength array in microns.
    signal_values : np.ndarray
        Signal arrays (input data).    
    baseline_type : str, {'polynomial', 'sinusoidal', 'spline'}
        Function type of fitted baseline.
    fitted_baseline_params : np.ndarray
        Fitted baseline parameters according to baseline_type.
    baseline_degree : int, optional
        Degree of fitted polynomial baseline. 
    """

    x = wavelength_values
    y = signal_values

    if baseline_type == 'polynomial':
        p = Polynomial(fitted_baseline_params)
        y_baseline = p(x)
        label = f"Fitted Polynomial Baseline (degree={baseline_degree})"
    elif baseline_type == 'sinusoidal':
        amplitude, freq, phase, offset = fitted_baseline_params
        y_baseline = amplitude * np.sin(2 * np.pi * freq * x + phase) + offset
        label = "Fitted Sinusoidal Baseline"
    elif baseline_type == 'spline':
        spline = fitted_baseline_params
        y_baseline = spline(x)
        label = "Fitted Spline Baseline"    
    else:
        raise ValueError(f"Invalid baseline_type '{baseline_type}'. Expected {{'polynomial', 'sinusoidal', 'spline'}}")

    # Create figure
    fig = plt.figure(figsize=(10, 6), dpi=700)
    # Create GridSpec object
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
    # Create first subplot for spectrum
    ax1 = fig.add_subplot(gs[:2,0])
    # Plot original spectrum
    ax1.plot(x, y, label="Original Spectrum", color="blue")
    # Plot fitted baseline
    ax1.plot(x, y_baseline, label=label, color="red", linestyle="--")
    # Create second subplot for residual
    ax2 = fig.add_subplot(gs[2, 0])
    y_residual = y - y_baseline
    # Plot residual 
    ax2.plot(x, y_residual, 
        label=f'Residual = (Data) - (Fitted {baseline_type.capitalize()} Baseline)',
        color='green')

    ax1.set_ylabel("Signal")
    ax2.set_xlabel("Wavelength [µm]")
    ax2.set_ylabel("Adjusted Signal")

    ax1.set_title(f"Spectra with Fitted {baseline_type.capitalize()} Baseline")
    
    ax1.legend()
    ax2.legend()

    plt.show()


def plot_baseline_fitting_bokeh(wavelength_values, 
                          signal_values, 
                          baseline_type, 
                          fitted_baseline_params, 
                          baseline_degree=None):
    """
    Plot the original spectrum and the fitted baseline using Bokeh.

    Parameters
    ----------
    wavelength_values : np.ndarray
        Wavelength array in microns.
    signal_values : np.ndarray
        Signal arrays (input data). 
    baseline_type : str, {'polynomial', 'sinusoidal', 'spline'}
        Function type of fitted baseline.
    fitted_baseline_params : np.ndarray
        Fitted baseline parameters according to baseline_type.
    baseline_degree : int, optional
        Degree of fitted polynomial baseline. 
    """

    x = wavelength_values
    y = signal_values

    if baseline_type == 'polynomial':
        p = Polynomial(fitted_baseline_params)
        y_baseline = p(x)
        baseline_label = f"Fitted Polynomial Baseline (degree={baseline_degree})"
    elif baseline_type == 'sinusoidal':
        amplitude, freq, phase, offset = fitted_baseline_params
        y_baseline = amplitude * np.sin(2 * np.pi * freq * x + phase) + offset
        baseline_label = "Fitted Sinusoidal Baseline"
    elif baseline_type == 'spline':
        spline = fitted_baseline_params
        y_baseline = spline(x)
        label = "Fitted Spline Baseline"    
    else:
        raise ValueError(f"Invalid baseline_type '{baseline_type}'. Expected {{'polynomial', 'sinusoidal', 'spline'}}")

    # Create the figure
    p1 = figure(title=f"Spectra with Fitted {baseline_type.capitalize()} Baseline",
               #x_axis_label="Wavelength [𝜇m]",
               y_axis_label="Signal",
               width=800, height=500,
               y_axis_type="linear",
               tools="pan,wheel_zoom,box_zoom,reset")

    # Add line plot
    p1.line(x, y, line_width=1.5, line_color='blue', alpha=0.8,
        legend_label="Original Spectrum")

    # Add line plot
    p1.line(x, y_baseline, line_width=3, line_color='red', line_dash='dashed', 
        legend_label=baseline_label)    

    # Create lower plot
    p2 = figure(title=' ',
               x_axis_label="Wavelength [𝜇m]",
               y_axis_label="Adjusted Signal",
               width=800, height=200,
               y_axis_type="linear",
               tools="pan,wheel_zoom,box_zoom,reset")

    # Add line plot
    residual = y - y_baseline
    p2.line(x, residual, line_width=1.5, line_color='green',
        legend_label=f'Residual = (Data) - (Fitted {baseline_type.capitalize()} Baseline)')  

    # Increase size of x and y ticks
    for p in (p1,p2):
        p.title.text_font_size = '14pt'
        p.xaxis.major_label_text_font_size = '14pt'
        p.xaxis.axis_label_text_font_size = '14pt'
        p.yaxis.major_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_size = '14pt'

    # Combine plots into a column
    layout = column(p1, p2)

    # Show the plot
    output_notebook()
    show(layout)





