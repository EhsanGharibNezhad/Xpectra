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

from typing import List, Union

import pprint

# ******* Data Visulaization Libraries ****************************
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MaxNLocator

from bokeh.plotting import figure, output_notebook, show
from bokeh.models import CustomJS, ColumnDataSource, TapTool, Div, HoverTool
from bokeh.layouts import column



def parse_line(line: str, columns: dict) -> dict:
    """
    Parse a single line of data according to the specified columns.
    
    Parameters
    ----------
    line : str
        Line of text to parse.
    columns : dict
        Dictionary defining column names, positions, data types, and format specifiers.
        
    Returns
    -------
    parsed_data : dict
        Dictionary where keys are column names and values are parsed data.
    """
    parsed_data = {}
    
    for key, (start, end, data_type, format_specifier) in columns.items():
        value_str = line[start:end].strip()
        try:
            if data_type == int:
                parsed_data[key] = int(value_str)
            elif data_type == float:
                parsed_data[key] = float(value_str)
            else:
                parsed_data[key] = value_str
        except ValueError:
            parsed_data[key] = None
    
    return parsed_data


def parse_file_to_dataframe(input_file: str, selected_columns: List[str]) -> pd.DataFrame:
    """
    Parse an input file into a pandas DataFrame based on specified columns.
    
    Parameters
    ----------
    input_file : str
        Path to the input file to parse.
    selected_columns : list
        List of column names to select from the input file.
        
    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing parsed data from the input file.
    """
    # Define the formats and the ranges for the columns
    columns = {
        'molec_id': (0, 2, int, "%2d"),
        'local_iso_id': (2, 3, int, "%1d"),
        'nu': (3, 15, float, "%12.6f"),
        'sw': (15, 25, float, "%10.3e"),
        'a': (25, 35, float, "%10.3e"),
        'gamma_air': (35, 40, float, "%5.4f"),
        'gamma_self': (40, 45, float, "%5.3f"),
        'elower': (45, 55, float, "%10.4f"),
        'n_air': (55, 59, float, "%4.2f"),
        'delta_air': (59, 67, float, "%8.6f"),
        'global_upper_quanta': (67, 82, str, "%15s"),
        'global_lower_quanta': (82, 97, str, "%15s"),
        'local_upper_quanta': (97, 112, str, "%15s"),
        'local_lower_quanta': (112, 127, str, "%15s"),
        'ierr': (127, 128, int, "%1d"),
        'iref': (128, 130, int, "%2d"),
        'line_mixing_flag': (130, 131, str, "%1s"),
        'gp': (131, 138, float, "%7.1f"),
        'gpp': (138, 145, float, "%7.1f")
    }

    # Filter columns based on selected_columns
    columns_to_use = {key: columns[key] for key in selected_columns}

    parsed_data_list = []  # Initialize an empty list to store parsed data dictionaries
    
    with open(input_file, 'r') as infile:
        for line in infile:
            data = parse_line(line, columns_to_use)
            parsed_data_list.append(data)  # Append each parsed line dictionary to the list
    
    df = pd.DataFrame(parsed_data_list)  # Convert list of dictionaries to DataFrame
    return df


def parse_hitran_description(hitran_description: str) -> pd.DataFrame:
    """
    Parse HITRAN data description and extract column information.

    Parameters
    ----------
    hitran_description : str
        description of HITRAN data fields.

    Returns
    -------
    df : pandas.DataFrame 
        DataFrame containing column name, format specifier, units, and description.
    """
    lines = hitran_description.strip().split('\n\n')
    data = []
    
    for line in lines:
        name = line.split('\n')[0].strip()
        format_specifier = None
        units = None
        description = None
        
        for subline in line.split('\n')[1:]:
            if 'C-style format specifier:' in subline:
                format_specifier = subline.split(': ')[1].strip()
            elif 'Units:' in subline:
                units = subline.split(': ')[1].strip()
            elif 'Description:' in subline:
                description = subline.split(': ')[1].strip()
        
        data.append({
            'Column Name': name,
            'Format Specifier': format_specifier,
            'Units': units,
            'Description': description
        })
    
    df = pd.DataFrame(data) 
    return df

def find_closest_data(hitran: pd.DataFrame, 
                      fitted_params: np.ndarray, 
                      weights: Union[list,np.ndarray] = None
                      ) -> pd.DataFrame:
    """
    Find the closest data points in the hitran DataFrame
    to multiple sets of fitted parameters, with weighted preference for earlier sets.

    Parameters
    ----------
    hitran : pd.DataFrame
        DataFrame with columns ['amplitude', 'center', 'wing'].
    fitted_params : np.ndarray
        2D array where each row contains [amplitude, center, width] for each fitted peak.
    weights : list or np.ndarray, optional
        List of weights for each set of fitted parameters. Default is None,
        which gives equal weight to each set.

    Returns
    -------
    closest_data_points : list of np.ndarray # CHANGE TO DATAFRAME
        List of arrays containing [amplitude, center, wing] of the closest data points
        in hitran DataFrame for each set of fitted parameters.
    """
    closest_data_points = []

    if weights is None:
        weights = np.ones(len(fitted_params))
    else:
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights to sum to 1

    for i, params in enumerate(fitted_params):
        # Calculate distances between each row in hitran and the current fitted parameter set
        distances = np.sqrt(np.sum((hitran[['nu', 'sw', 'gamma_air']].values - params)**2, axis=1))
        
        # Apply weight to the distances
        weighted_distances = distances * weights[i]

        # Find the index of the minimum weighted distance
        closest_index = np.argmin(weighted_distances)
        
        # Get the closest data point from hitran
        closest_data_point = hitran.iloc[closest_index].values

        closest_data_points.append(closest_data_point)

    return pd.DataFrame(closest_data_points, columns=hitran.columns)


def click_and_print(wavelength_values: np.ndarray,
                    signal_values: np.ndarray,
                    wavelength_range = Union[list, tuple, np.ndarray],
                    absorber_name: str = None,
                    y_label: str = "Signal",
                    title_label: str = None,
                    plot_type: str ='scatter'
                    ) -> None:
    """
    Click and print spectral peaks on plotted spectra with error bars using Bokeh.

    Parameters
    ----------
    wavelength_values : np.ndarray
        Wavelength array in microns.
    signal_values : np.ndarray
        Signal arrays (input data).
    wavelength_range : list-like, optional
        List-like object (list, tuple, or np.ndarray) with of length 2 representing wavelength range for plotting.
    absorber_name : str, optional
        Molecule or atom name.
    y_label : str, optional
        Label for the y-axis. Default is "Signal".
    title_label : str, optional
        Title of the plot. Default is None.
    plot_type : str, optional
        Type of plot. Can be 'scatter' or 'line'. Default is 'scatter'.
    """

    molecule_name = absorber_name
    x_obs = wavelength_values
    y_obs = signal_values

    # Trim x and y to desired wavelength range
    if wavelength_range is not None:
        # Make sure range is in correct format
        if len(wavelength_range) != 2:
            raise ValueError('wavelength_range must be tuple, list, or array with 2 elements')
        # Locate indices and splice
        condition_range = (x_obs > wavelength_range[0]) & (x_obs < wavelength_range[1])
        x_obs = x_obs[condition_range]
        y_obs = y_obs[condition_range]
    
    # Create a ColumnDataSource
    source = ColumnDataSource(data=dict(x=x_obs, y=y_obs))
    
    # Create the figure
    p = figure(title=f"{molecule_name}: Calibrated Laboratory Spectra" if title_label is None else title_label,
               x_axis_label="Wavelength [𝜇m]",
               y_axis_label=y_label,
               width=1000, height=300,
               y_axis_type="linear",
               tools="pan,wheel_zoom,box_zoom,reset")

    # Add HoverTool to the plot
    hover = HoverTool(tooltips=[("Wavelength", "@x{0.000} µm"), ("Signal", "@y{0.000}")], mode='vline')
    p.add_tools(hover)

    if plot_type == 'scatter':
        # Add the scatter plot
        p.scatter('x', 'y', source=source, size=4, fill_color='green', line_color=None, line_alpha=0.2,
                  legend_label=f"{molecule_name}: Laboratory Spectra")
    elif plot_type == 'line':
        # Add the line plot
        p.line('x', 'y', source=source, line_width=2, line_color='green', alpha=0.6,
               legend_label=f"{molecule_name}: Laboratory Spectra")

    # Increase size of x and y ticks
    p.title.text_font_size = '14pt'
    p.xaxis.major_label_text_font_size = '14pt'
    p.xaxis.axis_label_text_font_size = '14pt'
    p.yaxis.major_label_text_font_size = '14pt'
    p.yaxis.axis_label_text_font_size = '14pt'

    # Add a Div to display the results
    div = Div(text="Click to print coordinates", width=400, height=300)

    # JavaScript callback to save coordinates using hover tool data
    callback = CustomJS(args=dict(source=source, div=div), code="""
        // Initialize the global array if not already present
        if (!window.clickedCoordinates) {
            window.clickedCoordinates = [];
        }

        // Get the x and y coordinates of the click event
        const x = cb_obj.x;
        const y = cb_obj.y;

        // Find the closest data point in the source data
        const data = source.data;
        const x_data = data['x'];
        const y_data = data['y'];

        // Initialize the minimum distance and corresponding index
        let minDist = Infinity;
        let closestIndex = -1;

        // Iterate through the data to find the closest point
        for (let i = 0; i < x_data.length; i++) {
            const dist = Math.abs(x - x_data[i]);
            if (dist < minDist) {
                minDist = dist;
                closestIndex = i;
            }
        }

        // If a valid closest index is found, store the coordinates
        if (closestIndex !== -1) {
            const clickedX = x_data[closestIndex];
            const clickedY = y_data[closestIndex];
            window.clickedCoordinates.push({x: clickedX, y: clickedY});

            // Update the Div with the list of all coordinates
            const coordArray = window.clickedCoordinates.map(coord => `[${coord.x.toFixed(3)}, ${coord.y.toFixed(3)}]`).join(', ');
            div.text = `Clicked coordinates: [${coordArray}]`;
        }

        """)

    # Add the callback to the plot
    p.js_on_event('tap', callback)
    
    # Layout and show the plot
    layout = column(p, div)
    show(layout)



