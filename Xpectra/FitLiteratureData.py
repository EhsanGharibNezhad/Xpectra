import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from operator import itemgetter, attrgetter
from scipy.optimize import curve_fit
import pandas as pd
import bokeh as bk
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import seaborn as sns

import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from matplotlib.ticker import ScalarFormatter

from bokeh.io import output_notebook 
from bokeh.layouts import row, column
from bokeh.plotting import show,figure, output_file
from bokeh.models import ColumnDataSource, Whisker, CustomJS, Legend
from bokeh.palettes import Category10, Category20, Turbo256

from multiprocessing import Pool

import time

from .LineAssigner import *


TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628',
                  '#999999', '#e41a1c', '#dede00', '#984ea3']


class FitLiteratureData:
    """
    Fit and plot literature data.

    Parameters
    ----------
    literature_file : str, optional
        Excel spreadsheet containing literature data.
    hitran_file : str, optional
        File path containing HITRAN data.
    literature_df : pd.DataFrame, optional
        DataFrame containing literature data.
    """

    def __init__(self, 
                 literature_file = None,
                 hitran_file = None,
                 ):
        self.literature_file = literature_file
        self.hitran_file = hitran_file
        self.line_assigner_instance = LineAssigner(hitran_file = hitran_file) # Warning: from another module and could cause bugs

    def pb_excel_reader(self, sheet_name = None):
        """
        Converts pressure broadening data in an excel spreadsheet to a data frame. 
        """
        pb = pd.read_excel(self.literature_file, sheet_name = sheet_name)
        
        self.literature_df = pb

    @staticmethod
    def fit_Pbro_Pade(J, a0,a1,a2,a3,b1,b2,b3,b4): 
        X1 = a0 + (a1 *J) + (a2 *(J**2)) +  (a3* (J**3))
        X2 = 1 + (b1*J) + (b2 *(J**2)) +  (b3 *(J**3))+  (b4* (J**4))
        return (X1/X2)

    @staticmethod
    def fit_Pbro_Pade_shifted(J, a0,a1,a2,a3,b1,b2,b3,b4): 
        X1 = a0 + (a1 *J) + (a2 *(J**2)) +  (a3* (J**3))
        X2 = 1 + (b1*J) + (b2 *(J**2)) +  (b3 *(J**3))+  (b4* (J**4))
        min_=np.min(X1/X2)
        if min_<0:
            min_=-min_
        elif min_>=0:
            min_=0
        return (X1/X2)+min_

    @staticmethod
    def get_palette(num_categories):
        if num_categories <= 10:
            return Category10[10]
        elif num_categories <= 20:
            return Category20[20]
        else:
            # Use Turbo256 for more than 20 categories, and repeat colors if needed
            return Turbo256[:num_categories]

    @staticmethod
    def filter_dataframe(df, filters=None):
        """
        Filter the DataFrame based on multiple column value pairs or conditions.

        Parameters:
        df (pd.DataFrame): The DataFrame to filter.
        filters (dict): A dictionary where keys are column names and values are lists of values to filter by or conditions.

        Returns:
        pd.DataFrame: The filtered DataFrame.
        """
        if filters is None:
            return df
        
        # Start with the original DataFrame
        filtered_df = df.copy()
        
        # Apply each filter
        for col, condition in filters.items():
            if col in filtered_df.columns:
                if callable(condition):
                    filtered_df = filtered_df[condition(filtered_df[col])]
                else:
                    filtered_df = filtered_df[filtered_df[col].isin(condition)]
            else:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
        
        return filtered_df


    def parse_hitran_file(self, 
                        selected_columns = ['molec_id', 'local_iso_id', 
                        'nu', 'sw', 'gamma_air','gamma_self', 'n_air', 
                        'local_upper_quanta', 'local_lower_quanta']
                        ):
        """
        Using LineAssigner parse_file_to_dataframe, convert HITRAN file to DataFrame.
        """
        df = self.line_assigner_instance.parse_file_to_dataframe(selected_columns=selected_columns)

        self.hitran_df = df

        #return df

    @staticmethod
    def calculate_gamma_nT(J_low, sym_low, pb_coeffs):
        pbro = pb_coeffs
        if sym_low in ['F1', 'F2', 'A1', 'A2', 'E']:
            # Get the coefficients based on symmetry type
            coeffs_gamma = pbro[(pbro['sym_low'] == sym_low)&(pbro['coeff'] == 'gamma_L')].loc[:, 'a0':'b4'].values[0]
            gamma = FitLiteratureData.fit_Pbro_Pade(J_low, *coeffs_gamma)

            # Calculate gamma_H2 and gamma_He
            gamma_H2 = gamma
            gamma_He = 0.4 * gamma

            coeffs_nT = pbro[(pbro['sym_low'] == sym_low)&(pbro['coeff'] == 'n_T')].loc[:, 'a0':'b4'].values[0]
            n_T = FitLiteratureData.fit_Pbro_Pade(J_low, *coeffs_nT)

            # Optionally print the result
            # print(J_low, sym_low, gamma_H2, gamma_He)
            return gamma_H2, gamma_He, n_T
        else:
            return 0.05, 0.03, 0.5


    @staticmethod
    def modify_line(line, pb_coeffs):
        try:
            gamma_h2,gamma_he, n_T = np.round(FitLiteratureData.calculate_gamma_nT(
                                                    J_low = int(line[100:102]),
                                                    sym_low = line[102:104].replace(' ',''), 
                                                    pb_coeffs = pb_coeffs),5)
        except:
            gamma_h2,gamma_he, n_T  = 0.0500, 0.0300, 0.4

        modified_line = line.replace(
                line[35:40], str('%5.4f'%(gamma_h2)).lstrip('0')).replace(
                line[40:45], str('%5.3f'%(gamma_he))).replace(
                line[55:59], str('%4.2f'%(n_T)))
        
        return modified_line

    @staticmethod
    def replace_in_file(file_path, 
                        file_path_to_save,
                        pb_coeffs,
                        ):

        # Open the input file for reading
        with open(file_path, 'r') as infile, open(file_path_to_save, 'w') as outfile:
            for line in infile:
                # Modify line 
                modified_line = FitLiteratureData.modify_line(line, pb_coeffs)

                # Write the modified line to the output file
                outfile.write(modified_line)


    @staticmethod
    def replace_in_file_parallel(file_path, 
                        file_path_to_save,
                        pb_coeffs,
                        num_processes=4):

        # Read all lines from the file
        with open(file_path, 'r') as infile:
            lines = infile.readlines()

        # Create a pool of workers
        with Pool(processes=num_processes) as pool:
            # Distribute the work of processing lines in parallel
            modified_lines = pool.starmap(FitLiteratureData.modify_line, [(line, pb_coeffs) for line in lines])

        # Write all modified lines to the output file
        with open(file_path_to_save, 'w') as outfile:
            outfile.writelines(modified_lines)



    def plot_with_uncertainty(self,
                             x_fit = None,
                             param_to_fit = 'gamma_L [cm-1/atm]',
                             param_to_fit_uncertainty = 'gamma_uncertainty',
                             #num_iterations = 5,
                             #x_fit_interation_bound = [5,20],
                             param_to_sort = 'author',
                             include_authors = None,
                             filters = None, 
                             drop_na_authors = True, 
                             fit_4thPade = False,
                             print_fitted_params = True,
                             show_plot = True,
                             save_plot = False,
                             save_path = None,
                             ):
        
        """
        Create a scatter plot with uncertainty bars and optionally fit data using a 4th order Pade equation.

        Parameters:
        ----------
        df : pd.DataFrame, optional
            The input DataFrame containing columns 'J_low', 'gamma_L [cm-1/atm]', and 'author'.
        x_fit : list or None, optional 
            X values for the fit range. Defaults to None.
        param_to_sort : str, optional
            Name of column corresponding to parameter to color-code and label in legend.
        param_to_fit : str, optional
            Name of column corresponding to parameter to fit.
        param_to_fit_uncertainty : str, optional
            Name of column corresponding to uncertainty of parameter to fit.
        num_iterations : int, optional
            Number of fitting iterations. Defaults to 5.
        hidden_authors : list or None, optional
            Authors to hide from the plot. Defaults to None.
        filters : dict or None, optional
            Filters for the DataFrame. Defaults to None.
        drop_na_authors : bool, optional
            Drop rows with missing authors if True. Defaults to True.
        fit_4thPade : bool, optional
            Whether to fit the data using a 4th-order Pade equation. Defaults to True.
        show_plot : bool, optional
            Whether to display the plot immediately. Defaults to True.
        save_plot : bool, optional
            Whether to save the plot immediately. Defaults to False.
        save_path : str, optional
            Location to save plot. Defaults to None. 
        """

        df = self.literature_df

        required_columns = ['J_low', param_to_fit, param_to_fit_uncertainty, param_to_sort]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column.")

        # Drop rows with missing authors if specified
        if drop_na_authors:
            df = df.dropna(subset=['author']).copy()

        if include_authors:
            df = df[df['author'].isin(include_authors)].copy()
            
        # Apply any specified filters to the DataFrame
        if filters:
            df = self.filter_dataframe(df, filters)

        # Handle uncertainties or create a default range
        if 'gamma_uncertainty' in df.columns:
            df.loc[:, 'lower_bound'] = df[param_to_fit] - df[param_to_fit_uncertainty] / 2.
            df.loc[:, 'upper_bound'] = df[param_to_fit] + df[param_to_fit_uncertainty] / 2.
        else:
            df.loc[:, 'lower_bound'] = df[param_to_fit]
            df.loc[:, 'upper_bound'] = df[param_to_fit]
        
        # Create a color palette for different items in column
        column_contents = df[param_to_sort].dropna().unique()
        palette = self.get_palette(len(column_contents))
        color_map = dict(zip(column_contents, palette))
        
        # Filter out hidden authors
        # df_visible = df#[['author']] #[~df['author'].isin(hidden_authors)]
        # df['color'] = df['author'].map(color_map)
        # source_visible = ColumnDataSource(df)

        # Create the figure
        p = figure(title="Scatter Plot with Uncertainty", x_axis_label='J_low', y_axis_label=param_to_fit, 
                   width=900, height=500)

        # Plot each symmetry's data with a different color and add error bars
        for item in column_contents:
            
            column_data = df[df[param_to_sort] == item]
            source = ColumnDataSource(column_data)

            # Plot circles for the category
            circle_renderer = p.circle('J_low', param_to_fit, size=10, color=color_map[item], 
                legend_label=item, source=source, fill_alpha=0.6)

            # Add error bars (Whiskers)
            whisker = Whisker(source=source, base="J_low", upper="upper_bound", lower="lower_bound", 
                              level="glyph", line_width=2)
            whisker.upper_head.size = 8
            whisker.lower_head.size = 8
            p.add_layout(whisker)
            
            # Sync visibility of whiskers with the circles
            circle_renderer.js_on_change('visible', CustomJS(args=dict(whisker=whisker), code="""
            whisker.visible = cb_obj.visible;
            """))

        if fit_4thPade:
            
            # filter out NaN and avoid divide by 0
            id_good = ~np.isnan(df['J_low']) & ~np.isnan(df[param_to_fit]) & ~np.isnan(df[param_to_fit_uncertainty]) & ~(df[param_to_fit_uncertainty]==0)
            x = df['J_low'][id_good].copy().to_numpy()
            y = df[param_to_fit][id_good].copy().to_numpy()
            y_err = df[param_to_fit_uncertainty][id_good].copy().to_numpy()

            # if x_fit_interation_bound is None:
            #     popt, pcov = curve_fit(self.fit_Pbro_Pade, x, y, sigma=y_err, maxfev=50000)     
            # else:
            #     # trim to x_fit_interation_bound
            #     id_trim = np.where((x > x_fit_interation_bound[0]) & (x < x_fit_interation_bound[1]))[0]
            #     x_trim = x[id_trim]
            #     y_trim = y[id_trim]
            #     y_err_trim = y_err[id_trim]
            #     popt, pcov = curve_fit(self.fit_Pbro_Pade, x_trim, y_trim, sigma=y_err_trim, maxfev=50000)

            popt, pcov = curve_fit(self.fit_Pbro_Pade, x, y, sigma=y_err, maxfev=50000)     

            if x_fit is None:
                x_fit = np.arange(x.min(),x.max()+1)
            else:
                x_fit = np.arange(x_fit[0], x_fit[1]+1)

            y_fit = self.fit_Pbro_Pade(x_fit, *popt)

            line = p.line(x_fit, y_fit, line_color='blue', line_dash='dashed', 
                name='Fitted - 4th Pade Eq.')
            
            # Calculate the uncertainties (standard deviations of the parameters)
            perr = np.sqrt(np.diag(pcov))

            if print_fitted_params:
                param_labels = ['a0','a1','a2','a3','b1','b2','b3','b4']
                i=0
                for param, uncertainty in zip(popt, perr):
                    print(f"{param_labels[i]}: {round(param,3)} +/- {round(uncertainty,3)}")
                    i=i+1         
             
        # Retrieve, sort, & reassign legend items in alphebetical order
        legend_items = sorted(p.legend[0].items, key=lambda item: item.label['value'])
        p.legend[0].items = legend_items

        # Style the plot
        p.legend.title = param_to_sort.capitalize()
        p.legend.location = "bottom_left"
        p.legend.click_policy = "hide"  
        p.grid.grid_line_alpha = 0.4

        if show_plot:
            # Show the plot
            output_notebook()
            show(p)

        if save_plot:
            # Save the plot
            output_file(save_path)
            save(p)

        if fit_4thPade:
            # Return fitted 4thPade parameters
            return np.array(list(popt))  


    def plot_literature_hist(self,
                            hist_param = 'J_low',
                            hist_param_range = None,
                            sort_by = 'author',
                            filters = None,
                            bins = 15,
                            stat = 'count'):
        
        # Plot hist using seaborn hue to display distribution

        df = self.literature_df

        required_columns = [hist_param, sort_by]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column.")

        # Apply any specified filters to the DataFrame
        if filters:
            df = self.filter_dataframe(df, filters)

        # Create figure
        fig = plt.figure(dpi=600)
        sns.histplot(data=df, x=hist_param, hue=sort_by, element='step', stat=stat, 
            common_norm=False, bins=bins, alpha=0.5)

        if hist_param_range:
            plt.xlim(*hist_param_range)

        plt.show()



