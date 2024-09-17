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
from bokeh.plotting import show,figure
from bokeh.models import ColumnDataSource, Whisker, CustomJS, Legend
from bokeh.palettes import Category10, Category20, Turbo256


TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628',
                  '#999999', '#e41a1c', '#dede00', '#984ea3']


class FitLitData:
    """
    Fit and plot literature data.

    Parameters
    ----------
    literature_file : str, optional
        Excel spreadsheet containing literature data.
    literature_df : pd.DataFrame, optional
        DataFrame containing literature data.
    """

    def __init__(self, literature_file):
        self.literature_file = literature_file


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


    def plot_by_category(self,
                         category = 'author',
                         hidden_authors=None,
                         filters=None, 
                         dropNAN_authors = True, 
                         fit_4thPade = True,
                         __print__=False
                        ):
        
        """
        Plot gamma_L versus J_low color-coded by chosen category with interactive legend.

        Parameters
        ----------
        category : str, optional
            The category, corresponding to a literature_df column, to filter the plot by. Default is 'author'. 
        hidden_authors : list, optional
            List of authors to hide from plot. 
        filters : dict, optional
            Dictionary of filters 
        dropNAN_authors : bool, optional
            Default is True.
        fit_4thPade : bool, optional
            Fit and plot 4th order Pbro Pade equation to each item in category. Default is True.
        __print__ : bool, optional
            Default is False.
        """

        df = self.literature_df

        # Ensure the dataframe has all necessary columns
        if 'J_low' not in df.columns or 'gamma_L [cm-1/atm]' not in df.columns or 'author' not in df.columns:
            raise ValueError("DataFrame must contain 'J_low', 'gamma_L [cm-1/atm]', and 'author' columns.")
        
        if dropNAN_authors:
            df['author'].dropna(inplace=True)
            
        if hidden_authors is None:
            hidden_authors = []
            
        if filters:
            df = self.filter_dataframe(df, filters)
          
        # If uncertainty values are in the DataFrame (replace 'gamma_uncertainty' with the appropriate column name)
        if 'gamma_uncertainty' in df.columns:
            df['lower_bound'] = df['gamma_L [cm-1/atm]'] - df['gamma_uncertainty']/2.
            df['upper_bound'] = df['gamma_L [cm-1/atm]'] + df['gamma_uncertainty']/2.
        else:
            df['lower_bound'] = df['gamma_L [cm-1/atm]']
            df['upper_bound'] = df['gamma_L [cm-1/atm]']
        
        # Create a color palette for different items in category
        category_contents = df[category].dropna().unique()
        palette = self.get_palette(len(category_contents))
        color_map = dict(zip(category_contents, palette))

        # Filter the DataFrame to hide specific authors
        df_visible = df[~df['author'].isin(hidden_authors)]
        
        # Prepare the ColumnDataSource for visible authors
        df_visible['color'] = df_visible['author'].map(color_map)
        source_visible = ColumnDataSource(df_visible)

        # Create the figure
        p = figure(title="Scatter Plot with Uncertainty", x_axis_label='J_low', y_axis_label='gamma_L [cm-1/atm]', 
                   width=900, height=500)

        # Plot each symmetry's data with a different color and add error bars
        for item in category_contents:
            
            category_data = df[df[category] == item]
            source = ColumnDataSource(category_data)

            # Plot circles for the category
            circle_renderer = p.circle('J_low', 'gamma_L [cm-1/atm]', size=10, color=color_map[item], 
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
                
                popt, pcov = curve_fit(self.fit_Pbro_Pade, category_data['J_low'], category_data['gamma_L [cm-1/atm]'], 
                                      #sigma =df['gamma_uncertainty'], 
                                       maxfev=5000)

                x_fit = np.arange(category_data['J_low'].min(),category_data['J_low'].max()+1)
                y_fit = self.fit_Pbro_Pade(x_fit, popt[0], popt[1], popt[2], popt[3],popt[4], popt[5], popt[6], popt[7])
                # y_fit_err = compute_fit_uncertainty(x_fit, popt, pcov)

                line = p.line(x_fit, y_fit, 
                       line_color=color_map[item], line_dash='dashed', name='Fitted - 4th Pade Eq.')
                
                # Sync visibility of lines with the circles
                circle_renderer.js_on_change('visible', CustomJS(args=dict(line=line), code="""
                line.visible = cb_obj.visible;
                """))
        
        # Retrieve, sort, & reassign legend items in alphebetical order
        legend_items = sorted(p.legend[0].items, key=lambda item: item.label['value'])
        p.legend[0].items = legend_items

        # Style the plot
        p.legend.title = category.capitalize()
        p.legend.location = "bottom_left"
        p.legend.click_policy = "hide"  # Allows clicking on the legend to hide/show data for specific authors
        p.grid.grid_line_alpha = 0.5

        if __print__:
            print(zip())

        # Show the plot
        output_notebook()
        show(p)




