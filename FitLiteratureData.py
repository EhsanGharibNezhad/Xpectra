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

output_notebook()


TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
]

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628',
                  '#999999', '#e41a1c', '#dede00', '#984ea3']
