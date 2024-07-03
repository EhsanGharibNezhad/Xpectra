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



