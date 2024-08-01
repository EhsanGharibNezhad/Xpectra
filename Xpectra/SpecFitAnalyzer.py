import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Needed for wavelet decomposition
import pywt
from scipy.sparse.linalg import spsolve, splu

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.tools import TapTool
from bokeh.io import push_notebook
from bokeh.models import HoverTool, ColumnDataSource

# Nedded for ALS
from scipy import sparse
from scipy.linalg import cholesky
from scipy.sparse.linalg import spsolve
# Module for performing detailed spectral analysis, including feature extraction, peak identification, and line fitting.


# Import libraries
import numpy as np
import pandas as pd
import os
import pprint
from scipy.interpolate import interp1d, RegularGridInterpolator, UnivariateSpline
from scipy import stats, optimize
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.stats import chi2
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import List, Union
from numpy.polynomial import Polynomial

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib import rcParams
# from bokeh.plotting import output_notebook, figure, show
from bokeh.models import ColumnDataSource


from .SpecStatVisualizer import *

# add __plot__ and __print__ to all 


# Import local module
# from io_funs import LoadSave

class SpecFitAnalyzer:
    """
    Perform various tasks to process the lab spectra, including:

    - Load the data
    - Convert the units
    - Visualize the data
    - Label the quantum assignments

    Parameters
    ----------
    signal_values : np.ndarray, optional
        Signal arrays (input data).
    wavelength_names : List[str], optional
        Names of wavelengths in microns.
    wavelength_values : np.ndarray, optional
        Wavelength array in microns.
    absorber_name : str, optional
        Molecule or atom name.
    """

    def __init__(
            self,
            signal_values: Union[np.ndarray, None] = None,
            wavelength_names: Union[List[str], None] = None,
            wavelength_values: Union[np.ndarray, None] = None,
            absorber_name: Union[str, None] = None,
    ):
        self.signal_values = signal_values
        self.wavelength_names = wavelength_names
        self.wavelength_values = wavelength_values
        self.absorber_name = absorber_name


    def gaussian(self, x, center, amplitude, width):
        """
        Gaussian function.
        """
        return amplitude * np.exp(-(x - center) ** 2 / (2 * width ** 2))

    def lorentzian(self, x, center, amplitude, width):
        """
        Lorentzian function.
        """
        return amplitude / (1 + ((x - center) / width) ** 2)

    def voigt(self, x, center, amplitude, wid_g, wid_l):
        """
        Voigt profile function.
        """
        sigma = wid_g / np.sqrt(2 * np.log(2))
        gamma = wid_l / 2
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(wofz(z)).astype(float) / (sigma * np.sqrt(2 * np.pi))

    def fit_spectrum(self,
                     initial_guesses,
                     line_profile='gaussian',
                     fitting_method='lm',
                     __plot__=True,
                     __print__=True):
        """
        Fit a spectrum with multiple peaks using specified line profiles (gaussian, lorentzian, voigt).

        Parameters
        ----------
        initial_guesses : list
            List of initial guesses for parameters of the line profile.
        line_profile : str, {'gaussian', 'lorentzian', 'voigt'}, optional
            Type of line profile to use for fitting. Default is 'gaussian'.
        __plot__ : bool
            True or False.
        __print__ : bool
            True or False.

        Returns
        -------
        fitted_params : list
            List of fitted parameters for each peak.
        """
        x = self.wavelength_values
        #         y = self.signal_values if y!= None
        y = self.y_baseline_corrected

        fitted_params = []
        covariance_matrices = []

        if line_profile == 'gaussian':
            line_profile_func = self.gaussian
        elif line_profile == 'lorentzian':
            line_profile_func = self.lorentzian
        elif line_profile == 'voigt':
            line_profile_func = self.voigt
        else:
            raise ValueError(f"Unknown line profile: {line_profile}")

        for guess in initial_guesses:
            params, cov_matrix = curve_fit(line_profile_func, x, y, p0=guess, method=fitting_method, maxfev=1000000)
            fitted_params.append(params)
            covariance_matrices.append(cov_matrix)

        self.fitted_params = fitted_params
        self.covariance_matrices = covariance_matrices
        self.covariance_matrices = covariance_matrices
        # FIND ME

        if __plot__ == True:
            self.plot_fitted_spectrum_bokeh(line_profile=line_profile)

        if __print__ == True:
            self.print_fitted_parameters_df()

    #         return fitted_params

    # def fit_baseline(self):
    #     """
    #     Fit a sinusoidal baseline to the spectrum.
    #
    #     Returns
    #     -------
    #     params : list
    #         Fitted parameters [amplitude, frequency, phase, offset] of the sinusoidal baseline.
    #     """
    #     x = self.wavelength_values
    #     y = self.signal_values
    #
    #     def sine_wave(x, amplitude, freq, phase, offset):
    #         return amplitude * np.sin(2 * np.pi * freq * x + phase) + offset
    #
    #     # Initial guesses for amplitude, frequency, phase, and offset
    #     initial_guesses = [baseline_amplitude, baseline_frequency, 0, 0]
    #     params, _ = curve_fit(sine_wave, x, y, p0=initial_guesses, maxfev=1000000)
    #
    #     return params


    @staticmethod
    def rmse(predictions, targets):
        """
        Calculate root mean square error (RMSE).

        Parameters:
        - predictions: array-like, predicted values.
        - targets: array-like, true values.

        Returns:
        - RMSE value.
        """
        return np.sqrt(((predictions - targets) ** 2).mean())


    #         return df

    # def fit_polynomial_baseline(self, degree):
    #     """
    #     Fit a polynomial baseline to the spectrum using least squares.
    #
    #     Parameters
    #     ----------
    #     degree : int
    #         Degree of the polynomial to fit.
    #
    #     Returns
    #     -------
    #     params : np.ndarray
    #         Coefficients of the fitted polynomial.
    #     """
    #     x = self.wavelength_values
    #     y = self.signal_values
    #
    #     # Fit polynomial baseline using least squares
    #     p = Polynomial.fit(x, y, degree)
    #     self.fitted_baseline_params = p.convert().coef
    #     self.baseline_degree = degree
    #     return self.fitted_baseline_params


    def fit_polynomial_baseline(self, 
                                degree,
                                __plot_seaborn__ = False,
                                __plot_bokeh__ = False,
                                __print__ = False):
        """
        Fit a polynomial baseline to the spectrum using least squares.

        Parameters
        ----------
        degree : int
            Degree of the polynomial to fit.
        __plot_seaborn__ : bool
            True or False.
        __plot_bokeh__ : bool
            True or False.
        __print__ : bool
            True or False.

        Returns
        -------
        params : np.ndarray
            Coefficients of the fitted polynomial.
        """
        x = self.wavelength_values
        y = self.signal_values

        # Fit polynomial baseline using least squares
        p = Polynomial.fit(x, y, degree)
        self.fitted_baseline_params = p.convert().coef
        self.baseline_type = 'polynomial'
        self.baseline_degree = degree

        if __plot__:
            plot_baseline_fitting_seaborn(self.wavelength_values, self.signal_values, 
                self.baseline_type, self.fitted_baseline_params, 
                baseline_degree=self.baseline_degree)

        if __print__:
            print_results_fun(self.fitted_baseline_params, 
                print_title = f'Fitted Polynomial Baseline Coefficients (degree={self.baseline_degree})')

        return self.fitted_baseline_params


    def fit_sinusoidal_baseline(self, 
                                initial_guesses,
                                __plot_seaborn__ = False,
                                __plot_bokeh__ = False,
                                __print__ = False):
        """
        Fit a sinusoidal baseline to the spectrum.

        Parameters
        ----------
        initial_guesses : list
            Initial guesses for amplitude, frequency, phase, and offset.
        __plot_seaborn__ : bool
            True or False.
        __plot_bokeh__ : bool
            True or False.
        __print__ : bool
            True or False.

        Returns
        -------
        params : np.ndarray
            Parameters of the fitted sinusoidal baseline.
        """
        x = self.wavelength_values
        y = self.signal_values

        def sine_wave(x, amplitude, freq, phase, offset):
            return amplitude * np.sin(2 * np.pi * freq * x + phase) + offset

        params, _ = curve_fit(sine_wave, x, y, p0=initial_guesses, maxfev=1000000)
        self.fitted_baseline_params = params
        self.baseline_type = 'sinusoidal'
        
        if __plot_seaborn__:
            plot_baseline_fitting_seaborn(self.wavelength_values, self.signal_values, 
                self.baseline_type, self.fitted_baseline_params)
        if __plot_bokeh__:
            plot_baseline_fitting_bokeh(self.wavelength_values, self.signal_values, 
                self.baseline_type, self.fitted_baseline_params)
        if __print__:
            baseline_info = dict(zip(['Amplitude', 'Frequency', 'Phase', 'Offset'], 
                self.fitted_baseline_params))
            print_results_fun(baseline_info, 
                print_title = 'Fitted Sinusoidal Baseline Parameters')

        return self.fitted_baseline_params


    def fit_spline_baseline(self, 
                            s=None, 
                            __plot_seaborn__ = False,
                            __plot_bokeh__ = False):
        """
        Fit a spline baseline to the spectrum.

        Parameters
        ----------
        s : float or None
            Smoothing factor to control the spline smoothness. If None, it is set by the algorithm.
        __plot_seaborn__ : bool
            True or False.
        __plot_bokeh__ : bool
            True or False.

        Returns
        -------
        spline : UnivariateSpline
            The fitted spline object.
        """
        x = self.wavelength_values
        y = self.signal_values

        spline = UnivariateSpline(x, y, s=s)
        self.fitted_baseline_params = spline
        self.baseline_type = 'spline'

        if __plot_seaborn__:
            plot_baseline_fitting_seaborn(self.wavelength_values, self.signal_values, 
                self.baseline_type, self.fitted_baseline_params)
        if __plot_bokeh__:
            plot_baseline_fitting_bokeh(self.wavelength_values, self.signal_values, 
                self.baseline_type, self.fitted_baseline_params)

        return spline


    def als(self, lam=1e6, p=0.1, itermax=10, __plot__=True):
        r"""
        Implements an Asymmetric Least Squares Smoothing
        baseline correction algorithm (P. Eilers, H. Boelens 2005)

        Baseline Correction with Asymmetric Least Squares Smoothing
        based on https://web.archive.org/web/20200914144852/https://github.com/vicngtor/BaySpecPlots

        Baseline Correction with Asymmetric Least Squares Smoothing
        Paul H. C. Eilers and Hans F.M. Boelens
        October 21, 2005

        Description from the original documentation:

        Most baseline problems in instrumental methods are characterized by a smooth
        baseline and a superimposed signal that carries the analytical information: a series
        of peaks that are either all positive or all negative. We combine a smoother
        with asymmetric weighting of deviations from the (smooth) trend get an effective
        baseline estimator. It is easy to use, fast and keeps the analytical peak signal intact.
        No prior information about peak shapes or baseline (polynomial) is needed
        by the method. The performance is illustrated by simulation and applications to
        real data.


        Inputs:
            y:
                input data (i.e. chromatogram of spectrum)
            lam:
                parameter that can be adjusted by user. The larger lambda is,
                the smoother the resulting background, z
            p:
                wheighting deviations. 0.5 = symmetric, <0.5: negative
                deviations are stronger suppressed
            itermax:
                number of iterations to perform
        Output:
            the fitted background vector

        """
        x = self.wavelength_values
        y = self.signal_values

        L = len(y)
        D = sparse.eye(L, format='csc')
        D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
        D = D[1:] - D[:-1]
        D = D.T
        w = np.ones(L)
        for i in range(itermax):
            W = sparse.diags(w, 0, shape=(L, L))
            Z = W + lam * D.dot(D.T)
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)

        if __plot__:
            # Create a new plot with a title and axis labels
            p = figure(title="Raman Spectrum",
                       x_axis_label="Wavenumber (cm^-1)",
                       y_axis_label="Intensity",
                       width=800, height=400)

            # Add the original spectrum to the plot
            original_spectrum = p.line(x, y, legend_label="Original Spectrum", line_width=2, color="blue")

            # Add the baseline corrected spectrum to the plot
            corrected_spectrum = p.line(x, z, legend_label="Baseline correction with ALS", line_width=2, color="red")

            p.line(x, y - z, legend_label="Baseline correction with ALS", line_width=2, color="black")

            # Add HoverTool
            hover = HoverTool()
            hover.tooltips = [
                ("Wavenumber (cm^-1)", "@x"),
                ("Original Intensity", "@y"),
                ("Corrected Intensity", "@z"),
                ("Baseline Corrected Intensity", "@corrected_y")
            ]
            p.add_tools(hover)

            # Add HoverTool
            hover = HoverTool()
            hover.tooltips = [
                ("Wavenumber (cm^-1)", "@x{0.0000}"),
                ("Original Intensity", "@y{0.0000}"),
                ("Corrected Intensity", "@z{0.00}"),
                ("Baseline Corrected Intensity", "@corrected_y")
            ]
            p.add_tools(hover)

            # Show the results
            show(p)

        #         return z

    from scipy import sparse
    from scipy.sparse.linalg import spsolve, splu
    import numpy as np

    def arpls(self, lam=1e4, ratio=0.05, itermax=100, __plot__=True):
        r"""
        Baseline correction using asymmetrically
        reweighted penalized least squares smoothing
        Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
        Analyst, 2015, 140, 250 (2015)

        Abstract

        Baseline correction methods based on penalized least squares are successfully
        applied to various spectral analyses. The methods change the weights iteratively
        by estimating a baseline. If a signal is below a previously fitted baseline,
        large weight is given. On the other hand, no weight or small weight is given
        when a signal is above a fitted baseline as it could be assumed to be a part
        of the peak. As noise is distributed above the baseline as well as below the
        baseline, however, it is desirable to give the same or similar weights in
        either case. For the purpose, we propose a new weighting scheme based on the
        generalized logistic function. The proposed method estimates the noise level
        iteratively and adjusts the weights correspondingly. According to the
        experimental results with simulated spectra and measured Raman spectra, the
        proposed method outperforms the existing methods for baseline correction and
        peak height estimation.

        Inputs:
            y:
                input data (i.e. chromatogram of spectrum)
            lam:
                parameter that can be adjusted by user. The larger lambda is,
                the smoother the resulting background, z
            ratio:
                wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
            itermax:
                number of iterations to perform
        Output:
            the fitted background vector

        """
        y = self.signal_values
        N = len(y)
        D = sparse.eye(N, format='csc')
        D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
        D = D[1:] - D[:-1]

        H = lam * D.T @ D
        w = np.ones(N)
        for i in range(itermax):
            W = sparse.diags(w, 0)
            WH = W + H
            lu = splu(WH)  # Use sparse LU decomposition
            z = lu.solve(w * y)
            d = y - z
            dn = d[d < 0]
            m = np.mean(dn)
            s = np.std(dn)
            wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
            if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
                break
            w = wt

        self.y_baseline_corrected = y - z

        if __plot__:
            # Create a new plot with a title and axis labels
            p = figure(title="Laser Spectrum",
                       x_axis_label="Wavenumber (cm^-1)",
                       y_axis_label="Intensity",
                       width=800, height=400)

            # Add the original spectrum to the plot
            original_spectrum = p.line(self.wavelength_values, y, legend_label="Original Spectrum", line_width=2, color="blue")

            # Add the baseline corrected spectrum to the plot
            corrected_spectrum = p.line(self.wavelength_values, z, legend_label="Baseline correction with ARPLS", line_width=2, color="red")

            p.line(self.wavelength_values, y - z, legend_label="Baseline correction with ARPLS", line_width=2, color="black")
            # Show the results

            # Add HoverTool
            hover = HoverTool()
            hover.tooltips = [
                ("Wavenumber (cm^-1)", "@x{0.0000}"),
                ("Original Intensity", "@y{0.0000}"),
                ("Corrected Intensity", "@z{0.00}"),
                ("Baseline Corrected Intensity", "@corrected_y")
            ]
            p.add_tools(hover)

            show(p)

        #         return z
