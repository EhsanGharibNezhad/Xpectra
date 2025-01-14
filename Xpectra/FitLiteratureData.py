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
from typing import List, Union

from .LineAssigner import *


TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628',
                  '#999999', '#e41a1c', '#dede00', '#984ea3']


output_notebook()

class FitLiteratureData:
    """
    Fit and plot literature data.

    Parameters
    ----------    
    literature_file : str, optional
        Excel spreadsheet containing literature data.
    hitran_file : str, optional
        File path containing HITRAN data.
    __reference_data__ : str, optional
        Reference data path. 
    line_assigner_instance : Xpectra.LineAssigner, optional
        Instance of LineAssigner class for parsing linelist. 
    literature_df : pd.DataFrame, optional
        DataFrame containing literature data.
    hitran_df : pd.DataFrame, optional
        DataFrame containing parsed HITRAN linelist.
    """

    def __init__(self, 
                 literature_file = None,
                 hitran_file = None,
                 __reference_data__ = None
                 ):
        self.literature_file = literature_file
        self.hitran_file = hitran_file
        self.__reference_data__ = __reference_data__

    def pb_excel_reader(self, sheet_name):
        """
        Converts pressure broadening data in an excel spreadsheet to a data frame. 
        """
        literature_file = os.path.join(self.__reference_data__, 'datasets', sheet_name)

        pb = pd.read_excel(literature_file)
        
        self.literature_df = pb

    @staticmethod
    def fit_Pbro_Pade(J, a0,a1,a2,a3,b1,b2,b3,b4): 
        # 4th Pade equation. 
        X1 = a0 + (a1 *J) + (a2 *(J**2)) +  (a3* (J**3))
        X2 = 1 + (b1*J) + (b2 *(J**2)) +  (b3 *(J**3))+  (b4* (J**4))
        return (X1/X2)

    @staticmethod
    def fit_Pbro_Pade_shifted(J, a0,a1,a2,a3,b1,b2,b3,b4): 
        # Shifted 4th Pade equation. 
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
        # Get color palette based on number of categories. 
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

        Parameters
        ----------  
        df : pd.DataFrame)
            The DataFrame to filter.
        filters : dict
            A dictionary where keys are column names and values are lists of values to filter by or conditions.

        Returns
        -------
        filtered_df : pd.DataFrame
            The filtered DataFrame.
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
                        columns: Union[dict,None] = None,
                        ):
        """
        Using LineAssigner parse_file_to_dataframe, convert HITRAN file to DataFrame.
        """

        self.line_assigner_instance = LineAssigner(hitran_file = self.hitran_file) # Warning: from another module and could cause bugs
        
        if columns:
            df = self.line_assigner_instance.parse_file_to_dataframe(columns=columns)
        else: 
            df = self.line_assigner_instance.parse_file_to_dataframe()

        self.hitran_df = df

        #return df


    @staticmethod
    def calculate_gamma_nT(J_low: float, 
                            sym_low: str, 
                            pb_coeffs: pd.DataFrame
                            ) -> list:
        """
        Calculate gamma_H2, gamma_He, and n_T using fitted 4th pade coefficients for given symmetry and J values. 

        Parameters
        ----------  
        J_low : float
            Lower J number for calculation. 
        sym_low : str
            Lower symmetry for calculation. 
        pb_coeffs : pd.DataFrame
            DataFrame containing 4th pade coefficients. 

        Returns
        -------
        [gamma_H2, gamma_He, n_T] : list
            Calculated values.
        """
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

        local_lower_quanta = line[112:127].strip()
        J_low, sym_low, N_low = LineAssigner.parse_term_symbols([local_lower_quanta])[0]

        try:
            gamma_h2,gamma_he, n_T = np.round(FitLiteratureData.calculate_gamma_nT(
                                                    J_low = J_low,
                                                    sym_low = sym_low, 
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


    def replace_in_file_parallel(self,
                                pb_coeffs: pd.DataFrame,
                                save_name: str = 'output.txt',
                                num_processes: int = 8
                                ) -> None:

        file_path = self.hitran_file

        # Create path
        file_path_to_save = os.path.join(self.__reference_data__, 'outputs', save_name)

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
                             x_fit: Union[List, None] = None,
                             param_to_fit: str = 'gamma_L [cm-1/atm]',
                             param_to_fit_uncertainty: str = 'gamma_uncertainty',
                             #num_iterations = 5,
                             #x_fit_interation_bound = [5,20],
                             param_to_sort: str = 'author',
                             include_authors = None,
                             filters: Union[dict, None] = None, 
                             drop_na_authors: bool = True, 
                             fit_4thPade: bool = False,
                             print_fitted_params: bool = True,
                             show_plot: bool = True,
                             save_plot: bool = False,
                             save_path: str = None,
                             ):
        
        """
        Create a scatter plot with uncertainty bars and optionally fit data using a 4th order Pade equation.

        Parameters
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
            circle_renderer = p.scatter('J_low', param_to_fit, size=10, color=color_map[item], 
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
            show(p)

        if save_plot:
            # Save the plot
            output_file(save_path)
            save(p)

        if fit_4thPade:
            # Return fitted 4thPade parameters
            return np.array(list(popt))  


    def plot_literature_hist(self,
                            hist_param: str = 'J_low',
                            hist_param_range: Union[List, None] = None,
                            sort_by: str = 'author',
                            filters: Union[dict, None] = None,
                            bins: int = 15,
                            stat: str = 'count',
                            dpi: int = 400):
        
        """
        Plot histogram using seaborn hue of any data from literature, color-coded. 

        Parameters
        ---------- 
        hist_param : str, optional
            Name of DataFrame column for plotting. Default is 'J_low'. 
        hist_param_range: list, optional
            Range for x-axis of histogram. Default is None.
        sort_by : str, optional
            Name of DataFrame column for sorting. Default is 'author'. 
        filters : dict, optional
            Dictionary containing filters for literature DataFrame. 
        stat : str, optional
            Aggregate statistic to compute in each bin. Options are {'count', 
            'frequency', 'probability', 'percent', 'density'}. Default is 'count'.
        bins : int, optonal
            Number of bins. Default is 15.
        dpi: int, optional
            Resolution of the figure. Default is 400. 


        """
        df = self.literature_df

        required_columns = [hist_param, sort_by]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column.")

        # Apply any specified filters to the DataFrame
        if filters:
            df = self.filter_dataframe(df, filters)

        # Create figure
        fig = plt.figure(dpi=dpi)
        sns.histplot(data=df, x=hist_param, hue=sort_by, element='step', stat='count', 
            common_norm=False, bins=bins, alpha=0.5)


        # Set x-axis limits
        if not hist_param_range:
            hist_param_range = df[hist_param].min(), df[hist_param].max()
        
        plt.xlim(*hist_param_range)


        # Ensure x-axis shows integers only
        min_x, max_x = plt.xlim()
        plt.xticks(range(int(min_x), int(max_x) + 1))

        plt.show()


    def plot_fitted_value_seaborn(self, 
                          df_pbro: pd.DataFrame,
                          y_param: str = 'gamma_L',
                          J_range: list = [0,150],
                          __save_plot__: bool = False,
                          __show_plot__: bool = True
                          ):
        """
        Plot fitted n or gamma vs J_low, color-coded by symmetry


        Parameters
        ----------    
        df_pbro : pd.DataFrame
            DataFrame containing fitted coefficients.
        y_param : str, optional
            Parameter to plot on y-axis, options are {'gamma_L', 'n_T'}.
        J_range : list, optional
            J number range to plot. Default is [0,150].
        __save_plot__ : bool, optional
            Defaults to False.
        __show_plot__ : bool, optional
            Defaults to True.

        """

        J_arr = np.arange(*J_range)
        # df index
        indeces = np.where(df_pbro['coeff'] == y_param)[0]

        for index in indeces:
            coefficients = list(df_pbro.loc[index,'a0':'b4'].values)
            fitted_values = self.fit_Pbro_Pade(J_arr, *coefficients)
            plt.plot(J_arr, fitted_values, '.', label= df_pbro.loc[index,'sym_low'])

        plt.xlabel('J_low')
        plt.ylabel(f'Fitted {y_param}')
        plt.legend()
        
        if __save_plot__:

            # Assign file name
            save_file = f"fitted_{y_param}_4th_pade_results.pdf"

            plt.savefig(os.path.join(self.__reference_data__, 'figures', save_file), 
                dpi=700, bbox_inches='tight')
        
        if __show_plot__:
            plt.show()
        else:
            plt.clf()

    def plot_fitted_value_bokeh(self, 
                          df_pbro: pd.DataFrame,
                          y_param: str = 'gamma_L',
                          J_range: list = [0,150],
                          ):
        """
        Plot fitted n or gamma vs J_low, color-coded by symmetry

        Parameters
        ----------    
        df_pbro : pd.DataFrame
            DataFrame containing fitted coefficients.
        y_param : str, optional
            Parameter to plot on y-axis, options are {'gamma_L', 'n_T'}.
        J_range : list, optional
            J number range to plot. Default is [0,150].
        """

        J_arr = np.arange(*J_range)
        # df index
        indeces = np.where(df_pbro['coeff'] == y_param)[0]

        # Create figure
        p = figure(title=f'Fitted {y_param}', 
            x_axis_label='J_low', 
            y_axis_label=f'Fitted {y_param}',
            width=800, height=500,
            tools="pan,wheel_zoom,box_zoom,reset")

        # Call colors
        colors = Category10[10]

        for i, index in enumerate(indeces):
            coefficients = list(df_pbro.loc[index,'a0':'b4'].values)
            fitted_values = self.fit_Pbro_Pade(J_arr, *coefficients)
            label = df_pbro.loc[index,'sym_low']

            # Create source
            source = ColumnDataSource(data={'x': J_arr, 'y': fitted_values, 'label': [label] * len(J_arr)})

            color = colors[i % len(colors)]

            # Plot scatter
            p.scatter('x', 'y', marker='circle', source=source, legend_label=label, color=color, size=8)

        # Show the legend and display plot
        p.legend.title = 'Symbols'
        p.legend.location = 'top_left'
        show(p)

        





















    # @staticmethod 
    # def examine_line(line, filters):

    #     df = pd.DataFrame(line)

    #     # Apply any specified filters to the DataFrame
    #     if filters:
    #         df = self.filter_dataframe(df, filters)
        
    #     return modified_line

    # @staticmethod
    # def filter_file_parallel(file_path,
    #                         filters, 
    #                         num_processes=4):


    #     # Read all lines from the file
    #     with open(file_path, 'r') as infile:
    #         lines = infile.readlines()

    #     # Create a pool of workers
    #     with Pool(processes=num_processes) as pool:
    #         # Distribute the work of processing lines in parallel
    #         modified_lines = pool.starmap(FitLiteratureData.examine_line, [(line,filters) for line in lines])

    #     return modified_lines




