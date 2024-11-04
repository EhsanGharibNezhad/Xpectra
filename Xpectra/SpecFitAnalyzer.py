import matplotlib.pyplot as plt

# Needed for wavelet decomposition
import pywt
from scipy.sparse.linalg import spsolve, splu

# Nedded for ALS
from scipy import sparse
from scipy.linalg import cholesky
from scipy.sparse.linalg import spsolve, splu
from scipy.signal import find_peaks
# Module for performing detailed spectral analysis, including feature extraction, peak identification, and line fitting.

import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


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
from scipy.ndimage import gaussian_filter1d

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from typing import List, Union

from numpy.polynomial import Polynomial

from .SpecStatVisualizer import *


# Import local module
# from io_funs import LoadSave

def line_finder_auto(wavenumber_values: np.ndarray,
                    signal_values: np.ndarray,
                    wavenumber_range: Union[list, tuple, np.ndarray] = None,
                    sigma = 2,
                    __plot__ = True, # find me: add __print__ as well
                    ) -> None:
    """
    Click and print spectral peaks on plotted spectra with error bars using Bokeh.

    Parameters
    ----------
    wavenumber_values : np.ndarray
        Wavenumber array in cm^-1.
    signal_values : np.ndarray
        Signal arrays (input data).
    wavenumber_range : list-like, optional
        List-like object (list, tuple, or np.ndarray) with of length 2 representing wavenumber range for plotting.
    """

    x_obs = wavenumber_values
    y_obs = signal_values

    # Trim x and y to desired wavelength range
    if wavenumber_range is not None:
        # Make sure range is in correct format
        if len(wavenumber_range) != 2:
            raise ValueError('wavenumber_range must be tuple, list, or array with 2 elements')
        # Locate indices and splice
        condition_range = (x_obs > wavenumber_range[0]) & (x_obs < wavenumber_range[1])
        x_obs = x_obs[condition_range]
        y_obs = y_obs[condition_range]


    # smooth first!
    y_obs_smoothed = gaussian_filter1d(y_obs, sigma)
    
    peaks, info = find_peaks(y_obs_smoothed, height=0.05)

    if __plot__: 
        plot_auto_peaks_bokeh(x_obs, y_obs, peaks)

    peak_centers = x_obs[peaks[::-1]] # reverse peaks
    peak_heights = y_obs[peaks[::-1]]

    return peak_centers, peak_heights


def line_finder_manual(wavenumber_values: np.ndarray,
                    signal_values: np.ndarray,
                    wavenumber_range: Union[list, tuple, np.ndarray] = None, # find me - only np.ndarray
                    sigma = 2,
                    __plot__ = True, # find me - add __print__ parameter as well
                    ) -> None:
    """
    Click and print spectral peaks on plotted spectra with error bars using Bokeh.

    Parameters
    ----------
    wavenumber_values : np.ndarray
        Wavenumber array in cm^-1.
    signal_values : np.ndarray
        Signal arrays (input data).
    wavenumber_range : list-like, optional
        List-like object (list, tuple, or np.ndarray) with of length 2 representing wavenumber range for plotting.
    """

    x_obs = wavenumber_values
    y_obs = signal_values

    # Trim x and y to desired wavelength range
    if wavenumber_range is not None:
        # Make sure range is in correct format
        if len(wavenumber_range) != 2:
            raise ValueError('wavenumber_range must be tuple, list, or array with 2 elements')
        # Locate indices and splice
        condition_range = (x_obs > wavenumber_range[0]) & (x_obs < wavenumber_range[1])
        x_obs = x_obs[condition_range]
        y_obs = y_obs[condition_range]

    # smooth first!
    y_obs_smoothed = gaussian_filter1d(y_obs, sigma)

    if __plot__: 
        find_peaks_bokeh(x_obs, y_obs)

class SpecFitAnalyzer:
    """
    Perform various tasks to process the lab spectra, including:

    - Load/Save
    - Manipulate & Visualize
    - Label the quantum assignments

    Parameters
    ----------
    signal_values : np.ndarray, optional
        Signal arrays (input data).
    wavelength_names : List[str], optional
        Names of wavelengths in microns.
    wavenumber_values : np.ndarray, optional
        Wavenumber array in cm^-1.
    absorber_name : str, optional
        Molecule or atom name.
    """

    def __init__(
            self,
            signal_values: Union[np.ndarray, None] = None,
            wavelength_names: Union[List[str], None] = None,
            wavenumber_values: Union[np.ndarray, None] = None,
            absorber_name: Union[str, None] = None,
            __reference_data__: Union[str, None] = None,
            y_baseline_corrected: Union[np.ndarray, None] = None,
    ):
        self.signal_values = signal_values
        self.wavelength_names = wavelength_names
        self.wavenumber_values = wavenumber_values
        self.absorber_name = absorber_name
        self.__reference_data__ = __reference_data__
        self.y_baseline_corrected = y_baseline_corrected


    def gaussian(self, x: np.ndarray, center: float, amplitude: float, width: float) -> np.ndarray:
        """
        Gaussian function.
        """
        return amplitude * np.exp(-(x - center) ** 2 / (2 * width ** 2))

    def lorentzian(self, x: np.ndarray, center: float, amplitude: float, width: float) -> np.ndarray:
        """
        Lorentzian function.
        """
        return amplitude / (1 + ((x - center) / width) ** 2)

    def voigt(self, x: np.ndarray, center: float, amplitude: float, wid_g: float, wid_l: float) -> np.ndarray:
        """
        Voigt profile function.
        """
        sigma = wid_g / np.sqrt(2 * np.log(2))
        gamma = wid_l / 2
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(wofz(z)).astype(float) / (sigma * np.sqrt(2 * np.pi))

    def check_negative_nan(self) -> List[np.ndarray]:
        """
        Check the spectral data for negative or NAN values, and report their location.

        Parameters
        ----------        
        signal_values : np.ndarray, optional
            Signal arrays (input data).
        wavenumber_values : np.ndarray, optional
            Wavenumber array in cm^-1.   

        Returns
        -------  
        trimmed_spectrum : list
            List of 2 arrays containing x_trimmed and y_trimmed
        """

        x = self.wavenumber_values
        y = self.signal_values

        # Identify negative or nan values
        id_negative = np.where(y<0)[0]
        id_nan = np.where(np.isnan(y))[0]

        # Update instance attributes
        self.negative_indeces = id_negative
        self.nan_indeces = id_nan

        # Print results:
        
        # NAN
        if len(id_nan) == 0:
            print('No NAN values.')
        elif len(id_nan) != 0:
            N = len(id_nan)
            Ntot = len(x)
            fraction = N/Ntot
            print(f'{N} NAN values found ({fraction*100}% of data)')
        
        # Negative
        if len(id_negative) == 0:
            print('No negative values.')
        elif len(id_negative) != 0:
            N = len(id_negative)
            Ntot = len(x)
            fraction = N/Ntot
            print(f'{N} negative values found ({fraction*100:.2f}% of data)')

        # Delete any negative or NAN values
        id_delete = np.unique(np.concatenate((id_negative, id_nan)))
        x_cleaned = np.delete(x, id_delete)
        y_cleaned = np.delete(y, id_delete)
        
        cleaned_spectrum = [x_cleaned, y_cleaned]

        # Update instance attributes
        self.x_cleaned = x_cleaned
        self.y_cleaned = y_cleaned

        # Return cleaned values 
        return cleaned_spectrum


    def fit_spectrum(self,
                     initial_guesses: Union[list, np.ndarray],
                     line_profile: str = 'gaussian',
                     fitting_method: str = 'lm',
                     wavenumber_range: Union[list, tuple, np.ndarray] = None,
                     __plot_bokeh__: bool = False,
                     __plot_seaborn__: bool = False,
                     __save_plots__: bool = False,
                     __print__: bool = False,
                     __show_plots__: bool = True,
                     ) -> None:
        """
        Fit a spectrum with multiple peaks using specified line profiles (gaussian, lorentzian, voigt).

        Parameters
        ----------
        initial_guesses : list or np.ndarray
            List of initial guesses for parameters of the line profile.
        line_profile : str, {'gaussian', 'lorentzian', 'voigt'}, optional
            Type of line profile to use for fitting. Default is 'gaussian'.
        wavenumber_range : list-like, optional
            List-like object (list, tuple, or np.ndarray) with of length 2 representing wavenumber range to fit within.
        __plot_bokeh__ : bool
            True or False.
        __plot_seaborn__ : bool
            True or False.
        __print__ : bool
            True or False.

        Returns
        -------
        fitted_params : list
            List of fitted parameters for each peak.
        """

        # Define x 
        x = self.wavenumber_values

        # Define y
        try:
            y = self.y_baseline_corrected
        except AttributeError:
            logging.warning("'y_baseline_corrected' attribute does not exist. Using baseline-included 'signal_values' attribute instead.")
            y = self.signal_values

        # Check x and y are not set to default 'None'
        if x is None or y is None:
            logging.critical("Class initialized without necessary attributes, 'wavenumber_values' and 'signal_values'. Please assign them.")
            return

        # Trim x and y to desired wavelength range for plotting
        if wavenumber_range is not None:
            # Make sure range is in correct format
            if not isinstance(wavenumber_range, (list, tuple, np.ndarray)) or len(wavenumber_range) != 2:
                logging.critical("'wavenumber_range' must be tuple, list, or array with 2 elements.")
                return
            # Locate indices and splice
            condition_range = (x > wavenumber_range[0]) & (x < wavenumber_range[1])
            x = x[condition_range]
            y = y[condition_range]

        
        # error: range error with initial guesslines


        # Define line_profile_func
        if line_profile == 'gaussian':
            line_profile_func = self.gaussian
        elif line_profile == 'lorentzian':
            line_profile_func = self.lorentzian
        elif line_profile == 'voigt':
            line_profile_func = self.voigt
        else:
            logging.critical(f"Unknown line profile: {line_profile}." +  " Please choose one: {'gaussian', 'lorentzian', 'voigt'}.")
            return

        fitted_params = []
        covariance_matrices = []

        # error for initial_guesses shape !
        for guess in initial_guesses:
            params, cov_matrix = curve_fit(line_profile_func, x, y, p0=guess, method=fitting_method, maxfev=1000000)
            fitted_params.append(params)
            covariance_matrices.append(cov_matrix)

        self.fitted_params = np.array(fitted_params)
        self.covariance_matrices = np.array(covariance_matrices)


        if __plot_bokeh__ == True:
            plot_fitted_spectrum_bokeh(x,y,fitted_params,
                line_profile=line_profile,
                fitting_method=fitting_method)
        
        if __plot_seaborn__ == True:
            plot_fitted_spectrum_seaborn(x,y,fitted_params,
                line_profile=line_profile,
                fitting_method=fitting_method,
                __save_plots__ = __save_plots__,
                __reference_data__ = self.__reference_data__,
                __show_plots__ = __show_plots__)

        if __print__ == True:
            
            # Convert lists to arrays
            guess_arr = np.array(initial_guesses)
            fit_arr = np.array(fitted_params)

            # Create dictionary with fitted vs. guessed params 
            data = {
                'center_guess': guess_arr[:,0],
                'center_fit': fit_arr[:,0],
                'intensity_guess': guess_arr[:,1],
                'intensity_fit': fit_arr[:,1],
                'width_guess': guess_arr[:,2],
                'width_fit': fit_arr[:,2]
            }

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Show all rows
            pd.set_option('display.max_rows', None)

            display(df)
            #print_fitted_parameters_df(fitted_params,covariance_matrices)




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


    def fit_polynomial_baseline(self, 
                                degree: int,
                                __plot_seaborn__: bool = False,
                                __plot_bokeh__: bool = False,
                                __print__: bool = False
                                ) -> np.ndarray:
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
        x = self.wavenumber_values
        y = self.signal_values

        # Fit polynomial baseline using least squares
        p = Polynomial.fit(x, y, degree)
        self.fitted_baseline_params = p.convert().coef
        self.baseline_type = 'polynomial'
        self.baseline_degree = degree
        self.y_baseline_corrected_polynomial = y - p(x)

        if __plot_bokeh__:
            plot_baseline_fitting_bokeh(self.wavenumber_values, self.signal_values, 
                self.baseline_type, self.fitted_baseline_params, 
                baseline_degree=self.baseline_degree)

        if __plot_seaborn__:
            plot_baseline_fitting_seaborn(self.wavenumber_values, self.signal_values, 
                self.baseline_type, self.fitted_baseline_params, 
                baseline_degree=self.baseline_degree)

        if __print__:
            print_results_fun(self.fitted_baseline_params, 
                print_title = f'Fitted Polynomial Baseline Coefficients (degree={self.baseline_degree})')

        return self.fitted_baseline_params


    def fit_sinusoidal_baseline(self, 
                                initial_guesses: list,
                                __plot_seaborn__: bool = False,
                                __plot_bokeh__: bool = False,
                                __print__: bool = False
                                ) -> np.ndarray:
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
        x = self.wavenumber_values
        y = self.signal_values

        def sine_wave(x, amplitude, freq, phase, offset):
            return amplitude * np.sin(2 * np.pi * freq * x + phase) + offset

        params, _ = curve_fit(sine_wave, x, y, p0=initial_guesses, maxfev=1000000)
        self.fitted_baseline_params = params
        self.baseline_type = 'sinusoidal'
        self.y_baseline_corrected_sinusoidal = y - sine_wave(x,*params)

        if __plot_seaborn__:
            plot_baseline_fitting_seaborn(self.wavenumber_values, self.signal_values, 
                self.baseline_type, self.fitted_baseline_params)
        if __plot_bokeh__:
            plot_baseline_fitting_bokeh(self.wavenumber_values, self.signal_values, 
                self.baseline_type, self.fitted_baseline_params)
        if __print__:
            baseline_info = dict(zip(['Amplitude', 'Frequency', 'Phase', 'Offset'], 
                self.fitted_baseline_params))
            print_results_fun(baseline_info, 
                print_title = 'Fitted Sinusoidal Baseline Parameters')

        return self.fitted_baseline_params


    def fit_spline_baseline(self, 
                            s: float = None, 
                            __plot_seaborn__: bool = False,
                            __plot_bokeh__: bool = False
                            ) -> UnivariateSpline:
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
        x = self.wavenumber_values
        y = self.signal_values

        spline = UnivariateSpline(x, y, s=s)
        self.fitted_baseline_params = spline
        self.baseline_type = 'spline'
        self.y_baseline_corrected_spline = y - spline(x)

        if __plot_seaborn__:
            plot_baseline_fitting_seaborn(self.wavenumber_values, self.signal_values, 
                self.baseline_type, self.fitted_baseline_params)
        if __plot_bokeh__:
            plot_baseline_fitting_bokeh(self.wavenumber_values, self.signal_values, 
                self.baseline_type, self.fitted_baseline_params)

        return spline



    def als(self, 
            lam: float = 1e6, 
            p: float = 0.1, 
            itermax: int = 10, 
            __plot__: bool = True,
            __save_plots__: bool = False,
            __print__: bool = False,
            ) -> None:

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
        x = self.wavenumber_values
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

        y_baseline_corrected_ALS = y - z
        self.y_baseline_corrected_ALS = y_baseline_corrected_ALS
        self.baseline_type = 'als'

        if __plot__:
            plot_fitted_als_seaborn(wavenumber_values=x, signal_values=y, fitted_baseline=z, 
                __reference_data__=self.__reference_data__, baseline_type = 'als', 
                __save_plots__=__save_plots__)

        if __print__:
            print('Fitting parameters...')
            print('Metrics...')


    def arpls(self, 
              lam: float = 1e4, 
              ratio: float = 0.05, 
              itermax: int = 100, 
              __plot__: bool = True,
              __save_plots__: bool = False,
              __print__: bool = False,
              ) -> None:

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
        x = self.wavenumber_values
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

        y_baseline_corrected_ARPLS = y - z
        self.y_baseline_corrected_ARPLS = y_baseline_corrected_ARPLS
        self.baseline_type = 'arpls'

        if __plot__:
            plot_fitted_als_seaborn(wavenumber_values=x, signal_values=y, fitted_baseline=z, 
                __reference_data__=self.__reference_data__, baseline_type = 'arpls', 
                __save_plots__=__save_plots__)

        if __print__:
            print('Fitting parameters...')
            print('Metrics...')
            
            








