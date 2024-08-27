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

from typing import List, Union, Any

# ******* Data Visulaization Libraries ****************************
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib import rcParams

from bokeh.plotting import output_notebook, figure, show
from bokeh.models import HoverTool, ColumnDataSource, Range1d, Span, Legend, Label, LinearAxis
from bokeh.layouts import column

import numpy as np
from numpy.polynomial import Polynomial

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


def print_results_fun(targets: Any, print_title: str = None) -> None:
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


def print_fitted_parameters(fitted_params: np.ndarray, 
                            covariance_matrices: np.ndarray
                            ) -> None:
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


def print_fitted_parameters_df(fitted_params: np.ndarray, 
                               covariance_matrices: np.ndarray
                               ) -> None:
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


def plot_spectra_errorbar_bokeh(wavenumber_values: np.ndarray,
                                signal_values: np.ndarray,
                                wavenumber_range: Union[list, tuple, np.ndarray] = None,
                                signal_values_err: np.ndarray = None,
                                absorber_name: str = None,
                                y_label: str = "Signal",
                                title_label: str = None,
                                data_type: str ='x_y_yerr',
                                plot_type: str ='scatter'
                                ) -> None:
    """
    Plot the spectra with error bars using Bokeh.

    Parameters
    ----------
    wavenumber_values : np.ndarray
        Wavenumber array in cm^-1.
    signal_values : np.ndarray
        Signal arrays (input data).
    wavenumber_range : list-like, optional
        List-like object (list, tuple, or np.ndarray) with of length 2 representing wavenumber range for plotting.
    signal_values_err : np.ndarray, optional
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
    x_obs = wavenumber_values
    y_obs = signal_values
    y_obs_err = signal_values_err

    # Trim x and y to desired wavelength range
    if wavenumber_range is not None:
        # Make sure range is in correct format
        if len(wavenumber_range) != 2:
            raise ValueError('wavenumber_range must be tuple, list, or array with 2 elements')
        # Locate indices and splice
        condition_range = (x_obs > wavenumber_range[0]) & (x_obs < wavenumber_range[1])
        x_obs = x_obs[condition_range]
        y_obs = y_obs[condition_range]

    # Create the figure
    p = figure(title=f"{molecule_name}: Calibrated Laboratory Spectra" if title_label is None else title_label,
               x_axis_label="Wavenumber [cm^-1]",
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

    # Add HoverTool
    hover = HoverTool()
    hover.tooltips = [
        ("Wavenumber [cm^-1]", "@x{0.0000}"),
        ("Intensity", "@y{0.0000}")
        ]
    p.add_tools(hover)

    # Add a secondary x-axis with the transformation 10^4/x
    x_transformed = [10**4 / x for x in x_obs]
    p.extra_x_ranges = {"x_transformed": Range1d(start=min(x_transformed), end=max(x_transformed))}

    # Create a new axis and map it to the new range
    p.add_layout(LinearAxis(x_range_name="x_transformed", axis_label="Wavelength [μm]"), 'above')

    # Show the plot
    output_notebook()
    show(p)


def plot_spectra_errorbar_seaborn(wavenumber_values: np.ndarray,
                                  signal_values: np.ndarray,
                                  wavenumber_range: Union[list, tuple, np.ndarray] = None,
                                  signal_values_err: np.ndarray = None,
                                  absorber_name: str = None,
                                  y_label: str = "Signal",
                                  title_label: str = None,
                                  data_type: str = 'x_y_yerr',
                                  plot_type: str = 'scatter'
                                  ) -> None:
    """
    Plot the spectra with error bars using Seaborn.

    Parameters
    ----------
    wavenumber_values : nd.array
        Wavenumber array in cm^-1.
    signal_values : nd.array
        Signal arrays (input data).
    wavenumber_range : list-like, optional
        List-like object (list, tuple, or np.ndarray) with of length 2 representing wavenumber range for plotting.
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
    x_obs = wavenumber_values
    y_obs = signal_values
    y_obs_err = signal_values_err

    # Trim x and y to desired wavelength range
    if wavenumber_range is not None:
        # Make sure range is in correct format
        if len(wavenumber_range) != 2:
            raise ValueError('wavenumber_range must be tuple, list, or array with 2 elements')
        # Locate indices and splice
        condition_range = (x_obs > wavenumber_range[0]) & (x_obs < wavenumber_range[1])
        x_obs = x_obs[condition_range]
        y_obs = y_obs[condition_range]

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
    ax2.set_xticklabels([f'{10 ** 4 / tick:.3f}' for tick in ax1.get_xticks()])
    ax2.set_xlabel(r"Wavelength [$\mu$m]", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_baseline_fitting_seaborn(wavenumber_values: np.ndarray, 
                                  signal_values: np.ndarray, 
                                  baseline_type: str, 
                                  fitted_baseline_params: np.ndarray, 
                                  baseline_degree: int = None
                                  ) -> None:
    """
    Plot the original spectrum and the fitted baseline using Seaborn.

    Parameters
    ----------
    wavenumber_values : np.ndarray
        Wavenumber array in cm^-1.
    signal_values : np.ndarray
        Signal arrays (input data).    
    baseline_type : str, {'polynomial', 'sinusoidal', 'spline'}
        Function type of fitted baseline.
    fitted_baseline_params : np.ndarray
        Fitted baseline parameters according to baseline_type.
    baseline_degree : int, optional
        Degree of fitted polynomial baseline. 
    """

    x = wavenumber_values
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
    fig = plt.figure(figsize=(8, 6), dpi=700)
    # Create GridSpec object
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    # Create first subplot for spectrum
    ax1 = fig.add_subplot(gs[0,0])
    # Plot original spectrum
    ax1.plot(x, y, label="Original Spectrum", color="blue",lw=0.9,alpha=0.8)
    # Plot fitted baseline
    ax1.plot(x, y_baseline, label=label, color="red")
    # Create second subplot for residual
    ax2 = fig.add_subplot(gs[1, 0])
    y_residual = y - y_baseline
    # Plot residual 
    ax2.plot(x, y_residual, 
        label=f'[Data] - [Fitted {baseline_type.capitalize()} Baseline]',
        color='green', lw=0.9, alpha=0.8)
    # Plot zero-point
    ax2.plot(x, np.zeros_like(x), label='Zero point', color="red",linestyle="--")

    for ax in (ax1,ax2):
        # Turn on grid lines with transparency
        ax.grid(True, alpha=0.5)
        # Set axes ticks, inwards
        ax.tick_params(axis='both', which='major', direction='in', length=6, width=1)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', direction='in', length=3.5, width=1)  

    ax1.set_ylabel("Signal")
    ax2.set_xlabel("Wavenumber [cm^-1]")
    ax2.set_ylabel("Baseline-Corrected Signal")

    ax1.set_title(f"Spectra with Fitted {baseline_type.capitalize()} Baseline")
    
    ax1.legend()
    ax2.legend()

    plt.show()


def plot_baseline_fitting_bokeh(wavenumber_values: np.ndarray, 
                                signal_values: np.ndarray, 
                                baseline_type: str, 
                                fitted_baseline_params: np.ndarray, 
                                baseline_degree: int = None
                                ) -> None:
    """
    Plot the original spectrum and the fitted baseline using Bokeh.

    Parameters
    ----------
    wavenumber_values : np.ndarray
        Wavenumber array in cm^-1.
    signal_values : np.ndarray
        Signal arrays (input data). 
    baseline_type : str, {'polynomial', 'sinusoidal', 'spline'}
        Function type of fitted baseline.
    fitted_baseline_params : np.ndarray
        Fitted baseline parameters according to baseline_type.
    baseline_degree : int, optional
        Degree of fitted polynomial baseline. 
    """

    x = wavenumber_values
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
        baseline_label = "Fitted Spline Baseline"    
    else:
        raise ValueError(f"Invalid baseline_type '{baseline_type}'. Expected {{'polynomial', 'sinusoidal', 'spline'}}")

    
    y_baseline_corrected = y - y_baseline

    # Create ColumnDataSource
    source_p1 = ColumnDataSource(data=dict(x=x, y=y, y_baseline=y_baseline))
    source_p2 = ColumnDataSource(data=dict(x=x, y_baseline_corrected=y_baseline_corrected))

    # Create a shared range object for consistent zoom
    x_min,x_max = np.min(x),np.max(x)
    x_padding = (x_max-x_min) * 0.05
    x_range = Range1d(start=x_min-x_padding, end=x_max+x_padding)

    # Create the figure
    p1 = figure(title=f"Spectra with Fitted {baseline_type.capitalize()} Baseline",
               y_axis_label="Signal",
               width=800, height=350,
               x_range=x_range, 
               y_axis_type="linear",
               tools="pan,wheel_zoom,box_zoom,reset")

    # Add line plot
    p1.line('x', 'y', line_width=1.5, line_color='blue', alpha=0.8,
        legend_label="Original Spectrum", source = source_p1)

    # Add line plot
    p1.line('x', 'y_baseline', line_width=2.5, line_color='red', 
        legend_label=baseline_label, source = source_p1)    

    # Create lower plot
    p2 = figure(title=' ',
               x_axis_label="Wavenumber [cm^-1]",
               y_axis_label="Baseline-Corrected Signal",
               width=800, height=350,
               x_range=x_range, 
               y_axis_type="linear",
               tools="pan,wheel_zoom,box_zoom,reset")

    # Add line plot
    p2.line('x', 'y_baseline_corrected', line_width=1.5, line_color='green',
        legend_label=f'[Data] - [Fitted {baseline_type.capitalize()} Baseline]',
        source = source_p2)  

    # Add line plot
    p2.line(x, np.zeros_like(x), line_width=2.5, line_color='red', line_dash='dashed', 
        legend_label='Zero point')   
    
    for p in (p1,p2):
        # Increase size of x and y ticks
        p.title.text_font_size = '14pt'
        p.xaxis.major_label_text_font_size = '14pt'
        p.xaxis.axis_label_text_font_size = '14pt'
        p.yaxis.major_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_size = '14pt'

    # Add HoverTool
    hover_p1 = HoverTool()
    hover_p1.tooltips = [
        ("Wavenumber [cm^-1]", "@x{0.000}"),
        ("Intensity", "@y{0.000}"),
        (f"Fitted {baseline_type.capitalize()} Baseline", "@y_baseline{0.000}"),
    ]
    hover_p2 = HoverTool()
    hover_p2.tooltips = [
        ("Wavenumber [cm^-1]", "@x{0.000}"),
        ("Baseline-Corrected Intensity", "@y_baseline_corrected{0.000}"),
    ]
    p1.add_tools(hover_p1)
    p2.add_tools(hover_p2)

    # Combine plots into a column
    layout = column(p1, p2)

    # Show the plot
    output_notebook()
    show(layout)


def plot_fitted_als_bokeh(wavenumber_values: np.ndarray, 
                          signal_values: np.ndarray,
                          fitted_baseline: np.ndarray,
                          baseline_type: str = 'als'
                          ) -> None:

    """
    Plot the original spectrum and the fitted baseline using Bokeh.

    Parameters
    ----------
    wavenumber_values : np.ndarray
        Wavenumber array in cm^-1.
    signal_values : np.ndarray
        Signal arrays (input data). 
    fitted_baseline : np.ndarray
        Fitted baseline using baseline_type. 
    baseline_type : str, {'als', 'arpls'}
        Function type of fitted baseline.

    """
    x = wavenumber_values
    y = signal_values
    y_baseline = fitted_baseline
    y_baseline_corrected = y - y_baseline

    # Create a shared range object for consistent zoom
    x_min,x_max = np.min(x),np.max(x)
    x_padding = (x_max-x_min) * 0.05
    x_range = Range1d(start=x_min-x_padding, end=x_max+x_padding)

    # Create ColumnDataSource
    source_p1 = ColumnDataSource(data=dict(x=x, y=y, y_baseline=y_baseline))
    source_p2 = ColumnDataSource(data=dict(x=x, y_baseline_corrected=y_baseline_corrected))

    # Create the figure
    p1 = figure(title=f"Spectra with Fitted {baseline_type.upper()} Baseline",
               y_axis_label="Signal",
               width=800, height=350,
               x_range=x_range, 
               y_axis_type="linear",
               tools="pan,wheel_zoom,box_zoom,reset")
    
    # Add line plot
    p1.line('x', 'y', line_width=1.5, line_color='blue', alpha=0.8,
        legend_label="Original Spectrum", source=source_p1)

    # Add line plot
    p1.line('x', 'y_baseline', line_width=2.5, line_color='red', 
        legend_label=f'Fitted {baseline_type.upper()} Baseline', source=source_p1)  

    # Create lower plot
    p2 = figure(title=' ',
               x_axis_label="Wavenumber [cm^-1]",
               y_axis_label="Baseline-Corrected Signal",
               width=800, height=350,
               x_range=x_range, 
               y_axis_type="linear",
               tools="pan,wheel_zoom,box_zoom,reset")
   
   # Add line plot
    p2.line('x', 'y_baseline_corrected', line_width=1.5, line_color='green',
        legend_label=f"Baseline correction with {baseline_type.upper()}",
        source=source_p2)  

    # Add line plot
    p2.line(x, np.zeros_like(x), line_width=2.5, line_color='red', line_dash='dashed', 
        legend_label='Zero point')   
    
    for p in (p1,p2):
        # Increase size of x and y ticks
        p.title.text_font_size = '14pt'
        p.xaxis.major_label_text_font_size = '14pt'
        p.xaxis.axis_label_text_font_size = '14pt'
        p.yaxis.major_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_size = '14pt'

    # Add HoverTool
    hover_p1 = HoverTool()
    hover_p1.tooltips = [
        ("Wavenumber [cm^-1]", "@x{0.000}"),
        ("Intensity", "@y{0.000}"),
        (f"Fitted {baseline_type.upper()} Baseline", "@y_baseline{0.000}"),
    ]
    hover_p2 = HoverTool()
    hover_p2.tooltips = [
        ("Wavenumber [cm^-1]", "@x{0.000}"),
        ("Baseline-Corrected Intensity", "@y_baseline_corrected{0.000}"),
    ]
    p1.add_tools(hover_p1)
    p2.add_tools(hover_p2)

    # Combine plots into a column
    layout = column(p1, p2)

    # Show the plot
    output_notebook()
    show(layout)


def plot_fitted_spectrum_bokeh(wavenumber_values: np.ndarray, 
                               signal_values: np.ndarray,
                               fitted_params: np.ndarray,
                               wavenumber_range: Union[list, tuple, np.ndarray] = None,
                               line_profile: str = 'gaussian',
                               fitting_method: str = 'lm'
                               ) -> None:
    """
    Plot the original spectrum and the fitted peaks using Bokeh.

    Parameters
    ----------
    wavenumber_values : np.ndarray
        Wavenumber array in cm^-1.
    signal_values : np.ndarray
        Signal arrays (input data). 
    fitted_params : list
        List of fitted parameters of the line profile.
    wavenumber_range : list-like, optional
        List-like object (list, tuple, or np.ndarray) with of length 2 representing wavenumber range for plotting.
    line_profile : str, {'gaussian', 'lorentzian', 'voigt'}, optional
        Type of line profile to use for fitting. Default is 'gaussian'.
    """

    x = wavenumber_values
    y = signal_values

    fitted_peak_positions = np.array(fitted_params)[:,0]

    # Calculate fitted y values
    y_fitted = np.zeros_like(x)
    for params in fitted_params:
        if line_profile == 'gaussian':
            center, amplitude, width = params
            y_fitted += amplitude * np.exp(-(x - center) ** 2 / (2 * width ** 2))
        elif line_profile == 'lorentzian':
            center, amplitude, width = params
            y_fitted += amplitude / (1 + ((x - center) / width) ** 2)
        elif line_profile == 'voigt':
            center, amplitude, wid_g, wid_l = params
            sigma = wid_g / np.sqrt(2 * np.log(2))
            gamma = wid_l / 2
            z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
            y_fitted += amplitude * np.real(wofz(z)).astype(float) / (sigma * np.sqrt(2 * np.pi))

    # Calculate residual
    residual = y - y_fitted

    # Create a shared range object for consistent zoom
    x_min,x_max = np.min(x),np.max(x)
    x_padding = (x_max-x_min) * 0.05
    x_range = Range1d(start=x_min-x_padding, end=x_max+x_padding)

    # Create ColumnDataSource
    source_p1 = ColumnDataSource(data=dict(x=x, y=y, y_fitted=y_fitted))
    source_p2 = ColumnDataSource(data=dict(x=x, residual=residual))

    # Calculate RMSE
    rmse_value = np.sqrt(((y_fitted - y) ** 2).mean())

   # Create a new plot with a title and axis labels
    p1 = figure(title=f"Spectra with Fitted {line_profile.capitalize()} Peaks - RMSE = {rmse_value:.2f}",
               y_axis_label="Signal",
               width=800, height=500,
               x_range=x_range, 
               y_axis_type="linear",
               tools="pan,wheel_zoom,box_zoom,reset")

    # Trim x and y to desired wavelength range
    if wavenumber_range is not None:
        # Make sure range is in correct format
        if len(wavenumber_range) != 2:
            raise ValueError('wavenumber_range must be tuple, list, or array with 2 elements')
        # Locate indices and splice
        condition_range = (x > wavenumber_range[0]) & (x < wavenumber_range[1])
        x = x[condition_range]
        y = y[condition_range]
        y_fitted = y_fitted[condition_range]

    # Add the original or baseline-corrected spectrum to the plot
    p1.line('x', 'y', legend_label="Spectrum", line_width=2, color="blue", source=source_p1)

    # Add the fitted peaks spectrum to the plot
    p1.line('x', 'y_fitted', legend_label=f'Fitted {line_profile.capitalize()}', line_width=2, 
        color="red", source=source_p1)
    
    # Create lower plot
    p2 = figure(title=' ',
               x_axis_label="Wavenumber [cm^-1]",
               y_axis_label="Residual",
               width=800, height=200,
               x_range=x_range, 
               y_axis_type="linear",
               tools="pan,wheel_zoom,box_zoom,reset")

    # Add line plot
    p2.line('x', 'residual', line_width=1.5, line_color='green',
        legend_label=f'Residual = (Data) - (Fitted {line_profile.capitalize()} Peaks)',
        source=source_p2)  

    for p in (p1,p2):
        for q in range(len(fitted_params)):

            # Plot fitted peak centers
            vline = Span(location=fitted_peak_positions[q], dimension='height', 
                line_color='red', line_alpha=0.8, line_width=2)
            p.add_layout(vline)

    # Manually add legend entries
    dummy_line1 = p1.line(fitted_peak_positions[0], y[0], legend_label="Fitted Peak Center", 
        line_color='red', line_alpha=0.8, line_width=1.5)
    
    # Increase size of x and y ticks
    for p in (p1,p2):
        p.title.text_font_size = '14pt'
        p.xaxis.major_label_text_font_size = '14pt'
        p.xaxis.axis_label_text_font_size = '14pt'
        p.yaxis.major_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_size = '14pt'

    
    # Add HoverTool
    hover_p1 = HoverTool()
    hover_p1.tooltips = [
        ("Wavenumber [cm^-1]", "@x{0.000}"),
        ("Intensity", "@y{0.000}"),
        (f"Fitted {line_profile.capitalize()}", "@y_fitted{0.000}"),
    ]
    hover_p2 = HoverTool()
    hover_p2.tooltips = [
        ("Wavenumber [cm^-1]", "@x{0.000}"),
        ("Residual", "@residual{0.000}"),
    ]
    p1.add_tools(hover_p1)
    p2.add_tools(hover_p2)

    # Combine plots into a column
    layout = column(p1, p2)

    # Show the plot
    output_notebook()
    show(layout)


def plot_fitted_spectrum_seaborn(wavenumber_values: np.ndarray, 
                                 signal_values: np.ndarray,
                                 fitted_params: np.ndarray,
                                 wavenumber_range: Union[list, tuple, np.ndarray] = None,
                                 line_profile: str = 'gaussian',
                                 fitting_method: str = 'lm'
                                 ) -> None:
    """
    Plot the original spectrum and the fitted peaks using Bokeh.

    Parameters
    ----------
    wavenumber_values : np.ndarray
        Wavenumber array in cm^-1.
    signal_values : np.ndarray
        Signal arrays (input data). 
    fitted_params : list
        List of fitted parameters of the line profile.
    wavenumber_range : list-like, optional
        List-like object (list, tuple, or np.ndarray) with of length 2 representing wavenumber range for plotting.
    line_profile : str, {'gaussian', 'lorentzian', 'voigt'}, optional
        Type of line profile to use for fitting. Default is 'gaussian'.
    """

    x = wavenumber_values
    y = signal_values

    fitted_peak_positions = np.array(fitted_params)[:,0]

    # Calculate fitted y values
    y_fitted = np.zeros_like(x)
    for params in fitted_params:
        if line_profile == 'gaussian':
            center, amplitude, width = params
            y_fitted += amplitude * np.exp(-(x - center) ** 2 / (2 * width ** 2))
        elif line_profile == 'lorentzian':
            center, amplitude, width = params
            y_fitted += amplitude / (1 + ((x - center) / width) ** 2)
        elif line_profile == 'voigt':
            center, amplitude, wid_g, wid_l = params
            sigma = wid_g / np.sqrt(2 * np.log(2))
            gamma = wid_l / 2
            z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
            y_fitted += amplitude * np.real(wofz(z)).astype(float) / (sigma * np.sqrt(2 * np.pi))

    # Calculate RMSE
    rmse_value = np.sqrt(((y_fitted - y) ** 2).mean())

    # Trim x and y to desired wavelength range for plotting
    if wavenumber_range is not None:
        # Make sure range is in correct format
        if len(wavenumber_range) != 2:
            raise ValueError('wavenumber_range must be tuple, list, or array with 2 elements')
        # Locate indices and splice
        condition_range = (x > wavenumber_range[0]) & (x < wavenumber_range[1])
        x = x[condition_range]
        y = y[condition_range]
        y_fitted = y_fitted[condition_range]

    # Create figure
    fig = plt.figure(figsize=(10, 6), dpi=700)
    # Create GridSpec object
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])

    # Create first subplot for spectrum
    ax1 = fig.add_subplot(gs[:2,0])
    # Plot original spectrum
    ax1.plot(x, y, label="Original Spectrum", color="blue")
    # Plot fitted peaks
    ax1.plot(x, y_fitted, label=f'Fitted {line_profile.capitalize()}', color="red")

    ax1.set_ylabel("Signal")
    ax1.set_title(f"Spectra with Fitted {line_profile.capitalize()} Peaks - RMSE = {rmse_value:.2f}")

    # Create second subplot for residual
    ax2 = fig.add_subplot(gs[2, 0])
    y_residual = y - y_fitted
    # Plot residual 
    ax2.plot(x, y_residual, 
        label=f'Residual = (Data) - (Fitted {line_profile.capitalize()} Peaks)',
        color='green')

    ax2.set_xlabel(r"Wavenumber [cm$^{-1}$]")
    ax2.set_ylabel("Residual")

    for ax in (ax1, ax2):
        for i in range(len(fitted_peak_positions)):
            # Plot fitted peak centers
            ax.axvline(x=fitted_peak_positions[i], color='r', alpha=0.8, lw=0.8)

    ax1.plot([], [], color='r', alpha=0.8, lw=0.8, label='Fitted Peak Center') # label
    
    # Set axes ticks, inwards
    for ax in (ax1,ax2):

        ax.grid(True, alpha=0.25)

        ax.tick_params(axis='both', which='major', direction='in', length=7, width=1.1)
        ax.tick_params(axis='x', direction='in', top=True)
        ax.tick_params(axis='y', direction='in', right=True)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.1)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='y', which='minor', direction='in', right=True)   

    ax1.legend()
    ax2.legend()

    plt.show()


def plot_assigned_lines_seaborn(wavenumber_values: np.ndarray, 
                                signal_values: np.ndarray, 
                                fitted_hitran: pd.DataFrame,
                                fitted_params: np.ndarray,
                                columns_to_print: Union[str, List[str]],
                                wavenumber_range: Union[list, tuple, np.ndarray] = None,
                                line_profile: str = 'gaussian',
                                fitting_method: str = 'lm',
                                absorber_name: str = None
                                ) -> None:

    """
    Plot the original spectrum and the assigned lines using Seaborn.

    Parameters
    ----------
    wavenumber_values : np.ndarray
        Wavenumber array in cm^-1.
    signal_values : np.ndarray
        Signal arrays (input data). 
    fitted_hitran : pd.DataFrame
        Dataframe containing information about the assigned spectral lines, including columns ['amplitude', 'center', 'wing'].
    fitted_params : np.ndarray
        Fitted parameters of peaks.
    columns_to_print : str or list
        Columns to print corresponding to line positions. 
    wavenumber_range : list-like, optional
        List-like object (list, tuple, or np.ndarray) with of length 2 representing wavenumber range for plotting.
    line_profile : str, {'gaussian', 'lorentzian', 'voigt'}, optional
        Type of line profile to use for fitting. Default is 'gaussian'.
    """

    # option for printing different information
    # add fitted spectrum

    x = wavenumber_values
    y = signal_values

    line_positions = fitted_hitran["nu"].to_numpy()
    print_columns = fitted_hitran[columns_to_print].to_numpy(dtype=str)
    fitted_peak_positions = fitted_params[:,0]

    # Calculate fitted y values
    y_fitted = np.zeros_like(x)
    for params in fitted_params:
        if line_profile == 'gaussian':
            center, amplitude, width = params
            y_fitted += amplitude * np.exp(-(x - center) ** 2 / (2 * width ** 2))
        elif line_profile == 'lorentzian':
            center, amplitude, width = params
            y_fitted += amplitude / (1 + ((x - center) / width) ** 2)
        elif line_profile == 'voigt':
            center, amplitude, wid_g, wid_l = params
            sigma = wid_g / np.sqrt(2 * np.log(2))
            gamma = wid_l / 2
            z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
            y_fitted += amplitude * np.real(wofz(z)).astype(float) / (sigma * np.sqrt(2 * np.pi))

    # Trim x and y to desired wavelength range for plotting
    if wavenumber_range is not None:
        # Make sure range is in correct format
        if len(wavenumber_range) != 2:
            raise ValueError('wavenumber_range must be tuple, list, or array with 2 elements')
        # Locate indices and splice
        condition_range = (x > wavenumber_range[0]) & (x < wavenumber_range[1])
        x = x[condition_range]
        y = y[condition_range]
        y_fitted = y_fitted[condition_range]

    # Create figure
    fig = plt.figure(figsize=(10, 6), dpi=700)
    # Create GridSpec object
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])

    # Create first subplot for spectrum
    ax1 = fig.add_subplot(gs[:2,0])

    # Plot original spectrum
    ax1.plot(x, y, color='k', alpha=0.8, label="Original Spectrum")
    # Plot fitted peaks
    ax1.plot(x, y_fitted, color='r', alpha=0.8, lw=0.8,
        label=f'Fitted {line_profile.capitalize()}')
    
    ax1.set_ylabel("Signal")
    ax1.set_title(f"{absorber_name}: Spectrum with Fitted {line_profile.capitalize()} Peaks")
        
    # Create second subplot for residual
    ax2 = fig.add_subplot(gs[2, 0])
    y_residual = y - y_fitted

    # Plot residual 
    ax2.plot(x, y_residual, color='k', label=f'Residual = (Data) - (Fitted {line_profile.capitalize()} Peaks)')

    ax2.set_xlabel("Wavenumber [cm$^{-1}$]")
    ax2.set_ylabel("Residual")
    
    # Plot assigned HITRAN lines
    for ax in (ax1, ax2):
        for q in range(len(line_positions)):

            # Plot assigned HITRAN lines
            ax.axvline(x=line_positions[q], color='b')  # Plot vertical lines

            # Plot fitted peak centers
            ax.axvline(x=fitted_peak_positions[q], color='r', alpha=0.8)

    ax1.plot([], [], color='r', alpha=0.8, label='Fitted Peak Center') # label     
    ax1.plot([], [], color='b', label='HITRAN Assignment: \n ' +  ' \n '.join(columns_to_print)) # label 

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.legend(loc='center left', bbox_to_anchor=(0, 1.2))

    # label  
    y_up=0
    y_range = ax1.get_ylim()
    y_diff = y_range[1]-y_range[0]
    for q in range(len(line_positions)):
        # Calculate label position

        y_up += y_diff / (len(line_positions)+1)

        # Retrieve the label
        label_q_list = list(print_columns[q])
        label_q_pad = [' ' + item for item in label_q_list]
        label_q = '\n'.join(label_q_pad)

        # Add the text label
        ax1.text(line_positions[q], np.min(y) + y_up, label_q, color='b', va="top")

    for ax in (ax1,ax2):
        # Turn on grid lines with transparency
        ax.grid(True, alpha=0.25)

        # Set axes ticks, inwards
        ax.tick_params(axis='both', which='major', direction='in', length=7, width=1.1)
        ax.tick_params(axis='x', direction='in', top=True)
        ax.tick_params(axis='y', direction='in', right=True)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1.1)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='y', which='minor', direction='in', right=True)   

    plt.tight_layout()

    plt.show()



def plot_assigned_lines_bokeh(wavenumber_values: np.ndarray, 
                              signal_values: np.ndarray, 
                              fitted_hitran: pd.DataFrame,
                              fitted_params: np.ndarray,
                              columns_to_print: Union[str, List[str]],
                              wavenumber_range: Union[list, tuple, np.ndarray] = None,
                              line_profile: str = 'gaussian',
                              fitting_method: str = 'lm',
                              absorber_name: str = None
                              ) -> None:

    """
    Plot the original spectrum and the fitted peaks using Bokeh.

    Parameters
    ----------
    wavenumber_values : np.ndarray
        Wavenumber array in cm^-1.
    signal_values : np.ndarray
        Signal arrays (input data). 
    fitted_hitran : pd.DataFrame
        Dataframe containing information about the assigned spectral lines, including columns ['amplitude', 'center', 'wing'].
    fitted_params : np.ndarray
        Fitted parameters of peaks.
    columns_to_print : str or list
        Columns to print corresponding to line positions. 
    wavenumber_range : list-like, optional
        List-like object (list, tuple, or np.ndarray) with of length 2 representing wavenumber range for plotting.
    line_profile : str, {'gaussian', 'lorentzian', 'voigt'}, optional
        Type of line profile to use for fitting. Default is 'gaussian'.
    """

    x = wavenumber_values
    y = signal_values

    line_positions = fitted_hitran["nu"].to_numpy()
    print_columns = fitted_hitran[columns_to_print].to_numpy(dtype=str)
    fitted_peak_positions = fitted_params[:,0]

    # Calculate fitted y values
    y_fitted = np.zeros_like(x)
    for params in fitted_params:
        if line_profile == 'gaussian':
            center, amplitude, width = params
            y_fitted += amplitude * np.exp(-(x - center) ** 2 / (2 * width ** 2))
        elif line_profile == 'lorentzian':
            center, amplitude, width = params
            y_fitted += amplitude / (1 + ((x - center) / width) ** 2)
        elif line_profile == 'voigt':
            center, amplitude, wid_g, wid_l = params
            sigma = wid_g / np.sqrt(2 * np.log(2))
            gamma = wid_l / 2
            z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
            y_fitted += amplitude * np.real(wofz(z)).astype(float) / (sigma * np.sqrt(2 * np.pi))

    # Trim x and y to desired wavelength range
    if wavenumber_range is not None:
        # Make sure range is in correct format
        if len(wavenumber_range) != 2:
            raise ValueError('wavenumber_range must be tuple, list, or array with 2 elements')
        # Locate indices and splice
        condition_range = (x > wavenumber_range[0]) & (x < wavenumber_range[1])
        x = x[condition_range]
        y = y[condition_range]
        y_fitted = y_fitted[condition_range]   

    # Calculate residual
    residual = y - y_fitted

    # Create a shared range object for consistent zoom
    x_min,x_max = np.min(x),np.max(x)
    x_padding = (x_max-x_min) * 0.05
    x_range = Range1d(start=x_min-x_padding, end=x_max+x_padding)

    # Create ColumnDataSource
    source_p1 = ColumnDataSource(data=dict(x=x, y=y, y_fitted=y_fitted))
    source_p2 = ColumnDataSource(data=dict(x=x, residual=residual))

   # Create a new plot with a title and axis labels
    p1 = figure(title=f"{absorber_name}: Spectrum with Fitted {line_profile.capitalize()} Peaks",
               y_axis_label="Signal",
               width=800, height=500,
               x_range = x_range,
               y_axis_type="linear",
               tools="pan,wheel_zoom,box_zoom,reset")

    # Add the original spectrum to the plot
    p1.line('x', 'y', legend_label="Original Spectrum", line_width=2, color="black", source=source_p1)

    # Add the fitted spectrum to the plot
    p1.line('x', 'y_fitted', legend_label=f'Fitted {line_profile.capitalize()}', line_width=1, 
        color="red", source=source_p1)
    
    # Create lower plot
    p2 = figure(title=' ',
               x_axis_label="Wavenumber [cm^-1]",
               y_axis_label="Residual",
               width=800, height=250,
               x_range = x_range,
               y_axis_type="linear",
               tools="pan,wheel_zoom,box_zoom,reset")

    # Add line plot
    p2.line('x', 'residual', line_width=1.5, line_color='black',
        legend_label=f'Residual = (Data) - (Fitted {line_profile.capitalize()} Peaks)',
        source=source_p2)  

    # Plot assigned HITRAN lines
    for p in (p1,p2):
        for q in range(len(line_positions)):

            # Plot assigned HITRAN lines
            vline = Span(location=line_positions[q], dimension='height', line_width=2,
                line_color='blue')
            p.add_layout(vline)

            # Plot fitted peak centers
            vline = Span(location=fitted_peak_positions[q], dimension='height', 
                line_color='red', line_alpha=0.8, line_width=2)
            p.add_layout(vline)

    # Manually add legend entries
    dummy_line1 = p1.line(fitted_peak_positions[0], y[0], legend_label="Fitted Peak Center", 
        line_color='red', line_alpha=0.8, line_width=1.5)
    dummy_line2 = p1.line(fitted_peak_positions[0], y[0], legend_label='HITRAN Assignment: \n ' +  ', '.join(columns_to_print), 
        line_width=1.5, line_color='blue')
   
    # Plot text of the assigned line data
    y_range = np.max(y)-np.min(y)
    y_diff = 1.2*y_range
    y_up = 0
    for q in range(len(line_positions)):
        
        # Calculate label position
        y_up += y_diff / (len(line_positions)+1)

        # Retrieve the label
        label_q_list = list(print_columns[q])
        label_q_pad = [' ' + item for item in label_q_list]
        label_q = '\n'.join(label_q_pad)

        # Add the text label
        label = Label(x=line_positions[q], y=np.min(y) + y_up, 
            text=label_q, text_color="blue", text_align="left", text_baseline="top")
        p1.add_layout(label)

    # Customize legends
    legend1 = p1.legend[0]
    legend1.location = 'top_left'
    p1.add_layout(legend1, 'above')

    legend2 = p2.legend[0]
    legend2.location = 'top_left'
    p2.add_layout(legend2, 'below')

    # Increase size of x and y ticks
    for p in (p1,p2):
        p.title.text_font_size = '14pt'
        p.xaxis.major_label_text_font_size = '14pt'
        p.xaxis.axis_label_text_font_size = '14pt'
        p.yaxis.major_label_text_font_size = '14pt'
        p.yaxis.axis_label_text_font_size = '14pt'

    # Add HoverTool
    hover_p1 = HoverTool()
    hover_p1.tooltips = [
        ("Wavenumber [cm^-1]", "@x{0.000}"),
        ("Intensity", "@y{0.000}"),
        (f"Fitted {line_profile.capitalize()}", "@y_fitted{0.000}"),
    ]
    hover_p2 = HoverTool()
    hover_p2.tooltips = [
        ("Wavenumber [cm^-1]", "@x{0.000}"),
        ("Residual", "@residual{0.000}"),
    ]
    p1.add_tools(hover_p1)
    p2.add_tools(hover_p2)

    # Combine plots into a column
    layout = column(p1, p2, sizing_mode="stretch_width", height_policy="min", margin=0)

    # Show the plot
    output_notebook()
    show(layout)




