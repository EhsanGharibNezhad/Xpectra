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

import re
import os

from typing import List, Union

import pprint

# ******* Data Visulaization Libraries ****************************
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MaxNLocator

from .SpecStatVisualizer import *



class LineAssigner:
    """
    Perform quantum assignments to fitted spectra. 

    Parameters
    ----------
    fitted_params : np.ndarray, optional
        Fitted parameters of spectral peaks with [center, amplitude, width].
    hitran_file : str, optional
        File path containing HITRAN data.
    absorber_name : str, optional
        Molecule or atom name.
    wavenumber_values : np.ndarray, optional
        Wavenumber array in cm^-1
    signal_values : np.ndarray, optional
        Signal arrays (input data)
    
    hitran_df : pd.DataFrame
        DataFrame with columns ['amplitude', 'center', 'wing'].
    fitted_hitran : pd.DataFrame

    """

    def __init__(
            self,
            fitted_params: Union[list, np.ndarray, None] = None,
            hitran_file: Union[str, None] = None,
            absorber_name: Union[str, None] = None,
            wavenumber_values: Union[np.ndarray, None] = None,
            signal_values: Union[np.ndarray, None] = None,
            ):

        self.wavenumber_values = wavenumber_values
        self.signal_values = signal_values
        self.fitted_params = fitted_params
        self.hitran_file = hitran_file
        self.absorber_name = absorber_name


    @staticmethod
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


    @staticmethod
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

    @staticmethod
    def parse_term_symbols(term_list: list) -> list:
        """
        Parse and seperate a multi-term column into multiple single-term columns. 
        
        Parameters
        ----------
        term_list : list
            List of text to parse.

        Returns
        -------
        parsed_terms : list
            List of parsed data.
        """ 
        parsed_terms = []
        
        for term in term_list:
            # Use regular expression to capture number, letters, and the final number separately
            match1 = re.match(r"(\d+)\s*([A-Z]\d?)\s*(\d{1,3})", term)
            match2 = re.match(r"(\d+)\s+(\d{1,3})\s*([A-Z]\d?)", term) # if ground state, sym at the end
            
            if match1:
                # Extract the matched groups and clean them
                J_value = int(match1.group(1))
                sym_value = match1.group(2).strip()
                N_value = int(match1.group(3).strip())
                parsed_terms.append([int(J_value), sym_value, int(N_value)])
            elif match2:
                # Extract the matched groups and clean them
                J_value = int(match2.group(1))
                N_value = int(match2.group(2).strip())
                sym_value = match2.group(3).strip()
                parsed_terms.append([int(J_value), sym_value, int(N_value)])
            else:
                parsed_terms.append([None,None,None])
        return parsed_terms


    def parse_file_to_dataframe(self, 
                                selected_columns: List[str] = ['local_iso_id','nu','sw','gamma_air','local_upper_quanta','ierr']
                                ) -> pd.DataFrame:
        """
        Parse an input file into a pandas DataFrame based on specified columns.
        
        Parameters
        ----------
        input_file : str
            Path to the input file to parse.
        selected_columns : list
            List of column names to select from the input file. Default is
            ['local_iso_id','nu','sw','gamma_air','local_upper_quanta']. 
            
        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing parsed data from the input file.
        """

        input_file = self.hitran_file

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
                data = self.parse_line(line, columns_to_use)
                parsed_data_list.append(data)  # Append each parsed line dictionary to the list
        
        df = pd.DataFrame(parsed_data_list)  # Convert list of dictionaries to DataFrame
        
        if 'local_lower_quanta' in df.columns: 
            # Seperate terms
            df[['J_low','sym_low','N_low']] = self.parse_term_symbols(df['local_lower_quanta'])
        if 'local_upper_quanta' in df.columns:
            # Seperate terms
            df[['J_up','sym_up','N_up']] = self.parse_term_symbols(df['local_upper_quanta'])
        
        self.hitran_df = df

        return df
        

    # Modified to just use peak centers, and as input argument 
    def hitran_line_assigner(self,
                             peak_centers,
                             ierr_weights: bool = True,
                             weights: Union[list,np.ndarray,None] = None,
                             columns_to_print: List[str] = ["nu", "local_upper_quanta"],
                             wavenumber_range: Union[list, tuple, np.ndarray, None] = None,
                             __print__: bool = False,
                             __plot_bokeh__: bool = False,
                             __plot_seaborn__: bool = False,
                             __save_plot__: bool = False,
                             __reference_data__: str = None):
        """
        Find the closest data points in the hitran DataFrame
        to multiple sets of fitted parameters, with weighted preference for earlier sets.

        Parameters
        ----------
        weights : list or np.ndarray, optional
            List of weights for each set of fitted parameters. Default is None,
            which gives equal weight to each set.
        columns_to_print: list, optional
            List of column names from HITRAN dataframe to display on assigned lines if 
            plotted. Default is ["nu", "local_upper_quanta"]. 
        wavenumber_range : list-like, optional
            List-like object (list, tuple, or np.ndarray) of length 2 representing 
            wavenumber range for plotting. Default is None. 
        __plot_bokeh__ : bool, optional
            Default is False.
        __plot_seaborn__ : bool, optional
            Default is False.

        Returns
        -------
        fitted_hitran : pd.DataFrame
            DataFrame containing the closest data points in hitran DataFrame for each 
            set of fitted parameters.
        """        

        plot_args = [self.wavenumber_values, self.signal_values] 
        if __plot_bokeh__ and any(arg is None for arg in plot_args):
            raise ValueError("All required attributes (wavenumber_values, signal_values) must have a value when __plot_bokeh__ is True.")
        if __plot_seaborn__ and any(arg is None for arg in plot_args):
            raise ValueError("All required attributes (wavenumber_values, signal_values) must have a value when __plot_seaborn__ is True.")

        if 'hitran_df' not in self.__dict__:
            raise AttributeError("The 'hitran_df' attribute is missing. Ensure that data is loaded by running the 'parse_file_to_dataframe() method.")
        hitran_df = self.hitran_df

        #fitted_params = self.fitted_params

        # Find closest data points 
        closest_data_points = []

        # Default ierr weights
        if ierr_weights:
            weights = hitran_df["ierr"]
        # If set to false, input weight array or None is used 
        else:
            if weights is None:
                weights = np.ones(len(peak_centers))
            else:
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize weights to sum to 1

        for i, center in enumerate(peak_centers):
            # Calculate distances between each row in hitran_df and the current fitted parameter set
            distances = np.sqrt((hitran_df['nu'].values - center)**2)
            
            # Apply weight to the distances
            weighted_distances = distances * weights[i]

            # Find the index of the minimum weighted distance
            closest_index = np.argmin(weighted_distances)
            
            # Get the closest data point from hitran_df
            closest_data_point = hitran_df.iloc[closest_index].values

            closest_data_points.append(closest_data_point)

        fitted_hitran = pd.DataFrame(closest_data_points, columns=hitran_df.columns)

        fitted_hitran['peak_centers'] = peak_centers
        # fitted_hitran['fitted_peak_amplitude'] = fitted_params[:,1]
        # fitted_hitran['fitted_peak_width'] = fitted_params[:,2]

        fitted_hitran = fitted_hitran[['peak_centers', 'nu', 'sw', 'gamma_air', 'local_iso_id', 'J_up', 'sym_up', 'N_up']]
        
        self.fitted_hitran = fitted_hitran
        
        if __plot_bokeh__:
            plot_hitran_lines_bokeh(self.wavenumber_values, self.signal_values, 
                                      fitted_hitran, peak_centers, columns_to_print = columns_to_print,
                                       wavenumber_range=wavenumber_range, absorber_name=self.absorber_name)
        if __plot_seaborn__:
            plot_hitran_lines_seaborn(self.wavenumber_values, self.signal_values, 
                                      fitted_hitran, peak_centers, columns_to_print = columns_to_print,
                                       wavenumber_range=wavenumber_range, absorber_name=self.absorber_name)
        if __save_plot__:
            plot_hitran_lines_seaborn(self.wavenumber_values, self.signal_values, 
                                      fitted_hitran, peak_centers, columns_to_print = columns_to_print,
                                       wavenumber_range=wavenumber_range, absorber_name=self.absorber_name,
                                       __save_plot__=True, __reference_data__=__reference_data__, __show_plot__=False)


        if __print__:
            display(fitted_hitran)

    # Original that needs fitted params {center, height, width} as instance attribute 
    # def hitran_line_assigner(self,
    #                          ierr_weights: bool = True,
    #                          weights: Union[list,np.ndarray,None] = None, 
    #                          columns_to_print: List[str] = ["nu", "local_upper_quanta"],
    #                          wavenumber_range: Union[list, tuple, np.ndarray, None] = None,
    #                          __print__: bool = False,
    #                          __plot_bokeh__: bool = False,
    #                          __plot_seaborn__: bool = False):
    #     """
    #     Find the closest data points in the hitran DataFrame
    #     to multiple sets of fitted parameters, with weighted preference for earlier sets.

    #     Parameters
    #     ----------
    #     weights : list or np.ndarray, optional
    #         List of weights for each set of fitted parameters. Default is None,
    #         which gives equal weight to each set.
    #     columns_to_print: list, optional
    #         List of column names from HITRAN dataframe to display on assigned lines if 
    #         plotted. Default is ["nu", "local_upper_quanta"]. 
    #     wavenumber_range : list-like, optional
    #         List-like object (list, tuple, or np.ndarray) of length 2 representing 
    #         wavenumber range for plotting. Default is None. 
    #     __plot_bokeh__ : bool, optional
    #         Default is False.
    #     __plot_seaborn__ : bool, optional
    #         Default is False.

    #     Returns
    #     -------
    #     fitted_hitran : pd.DataFrame
    #         DataFrame containing the closest data points in hitran DataFrame for each 
    #         set of fitted parameters.
    #     """        


    #     plot_args = [self.wavenumber_values, self.signal_values] 
    #     if __plot_bokeh__ and any(arg is None for arg in plot_args):
    #         raise ValueError("All required attributes (wavenumber_values, signal_values) must have a value when __plot_bokeh__ is True.")
    #     if __plot_seaborn__ and any(arg is None for arg in plot_args):
    #         raise ValueError("All required attributes (wavenumber_values, signal_values) must have a value when __plot_seaborn__ is True.")

    #     if 'hitran_df' not in self.__dict__:
    #         raise AttributeError("The 'hitran_df' attribute is missing. Ensure that data is loaded by running the 'parse_file_to_dataframe() method.")
    #     hitran_df = self.hitran_df

    #     fitted_params = self.fitted_params

    #     # Find closest data points 
    #     closest_data_points = []

    #     # Default ierr weights
    #     if ierr_weights:
    #         weights = hitran_df["ierr"]
    #     # If set to false, input weight array or None is used 
    #     else:
    #         if weights is None:
    #             weights = np.ones(len(fitted_params))
    #         else:
    #             weights = np.array(weights)
    #             weights = weights / np.sum(weights)  # Normalize weights to sum to 1

    #     for i, params in enumerate(fitted_params):
    #         # Calculate distances between each row in hitran_df and the current fitted parameter set
    #         distances = np.sqrt(np.sum((hitran_df[['nu', 'sw', 'gamma_air']].values - params)**2, axis=1))
            
    #         # Apply weight to the distances
    #         weighted_distances = distances * weights[i]

    #         # Find the index of the minimum weighted distance
    #         closest_index = np.argmin(weighted_distances)
            
    #         # Get the closest data point from hitran_df
    #         closest_data_point = hitran_df.iloc[closest_index].values

    #         closest_data_points.append(closest_data_point)

    #     fitted_hitran = pd.DataFrame(closest_data_points, columns=hitran_df.columns)

    #     fitted_hitran['fitted_peak_center'] = fitted_params[:,0]
    #     fitted_hitran['fitted_peak_amplitude'] = fitted_params[:,1]
    #     fitted_hitran['fitted_peak_width'] = fitted_params[:,2]

    #     fitted_hitran = fitted_hitran[['local_iso_id', 'nu', 'fitted_peak_center', 'sw', 'fitted_peak_amplitude', 'gamma_air', 'fitted_peak_width', 'local_upper_quanta']]
    #     self.fitted_hitran = fitted_hitran
        
    #     if __plot_bokeh__:
    #         plot_assigned_lines_bokeh(self.wavenumber_values, self.signal_values, 
    #                                   fitted_hitran, fitted_params, columns_to_print = columns_to_print,
    #                                   wavenumber_range=wavenumber_range, absorber_name=self.absorber_name)
    #     if __plot_seaborn__:
    #         plot_assigned_lines_seaborn(self.wavenumber_values, self.signal_values, 
    #                                     fitted_hitran, fitted_params, columns_to_print = columns_to_print,
    #                                     wavenumber_range=wavenumber_range, absorber_name=self.absorber_name)
    #     if __print__:
    #         display(fitted_hitran)




