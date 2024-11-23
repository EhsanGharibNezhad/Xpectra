
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Union, Any

from hapi import *  
from .SpecStatVisualizer import *





def generate_XS(linelist_name: list,
			    res: float,
			    P_bar: float,
			    T: float,  
			    WN1: float,
			    WN2: float,
			    Diluent: dict,
			    ind_1460: int,
			    Intensity_cutoff: float = 1e-39,  
			    line_profile: str = 'absorptionCoefficient_Voigt',
			    output_dic: str = 'output',
			    plot_bokeh: bool = False,
			    plot_seaborn: bool = False,
			    save: bool = True
				) -> None:
    """
    Generates cross-sections based on provided parameters.

    Parameters:
        linelist (list): Source tables for calculating the cross-section.
        res (float): Wavenumber resolution.
        P_bar (float): Pressure in bar.
        T (float): Temperature in Kelvin.
        WN1 (float): Initial wavenumber in cm⁻¹.
        WN2 (float): Final wavenumber in cm⁻¹.
        Diluent (dict): Gas mixture composition (e.g., {'h2': 0.85, 'he': 0.15} for hydrogen-dominant atmospheres).
        ind_1460 (int): Index used for saving files.
        Intensity_cutoff (float, optional): Intensity threshold for filtering. Default is 1e-39.
        plot_bokeh (bool, optional): If True, plot the absorption coefficients with bokeh. Default is False.
        plot_seaborn (bool, optional): If True, plot the absorption coefficients with seaborn. Default is False.
        save (bool, optional): If True, save the results to files. Default is True.

    Returns:
        tuple: (nu4, coef4), where `nu4` is the wavenumber array, and `coef4` is the absorption coefficient array.

    Notes:
        - The `Diluent` parameter is more flexible than the deprecated `GammaL` parameter.
        - If both `Diluent` and `GammaL` are specified, `GammaL` is ignored.
        - For pressures <= 200 bar, the wing cutoff is set to 25 cm⁻¹. For higher pressures, it is set to 50 cm⁻¹.
    """
    # Convert pressure from bar to atm
    P_atm = P_bar * (1.0E+05 / 1.01325E+05)  # Ref: NIST SI guide for conversion factors

    # Determine wing cutoff based on pressure
    Wing_cutoff = 25 if P_bar <= 200 else 50

    if line_profile == 'absorptionCoefficient_Voigt':

        # Calculate absorption coefficients using Voigt profile
        wn, coef = absorptionCoefficient_Voigt(
            SourceTables=linelist_name,       # List of source tables
            OmegaStep=res,              # Wavenumber resolution
            HITRAN_units=True,          # XS unit: cm2/molecule
            WavenumberWing=Wing_cutoff, # Line shape wing cutoff (in cm⁻¹)
            Environment={'p': P_atm, 'T': T},  # Pressure (atm) and Temperature (K)
            Diluent=Diluent             # Broadening mixture composition
        )
        
    else:
        raise ValueError("No valid line profile specified")

    
    if plot_bokeh:
        plot_spectra_errorbar_bokeh(wn, coef,wavenumber_range=(WN1,WN2),
                                    y_label='Absorption Cross Section',
                                    title_label = 'Cross Section vs Wavenumber',
                                    plot_type = 'line')
    if plot_seaborn:
        plot_spectra_errorbar_seaborn(wn, coef,wavenumber_range=(WN1,WN2),
                                    y_label='Absorption Cross Section',
                                    title_label = 'Cross Section vs Wavenumber',
                                    plot_type = 'line')
        
    # Save results if required
    if save:
        wn_path = os.path.join(output_dic,f"wn.{int(ind_1460)}")
        coef_path = os.path.join(output_dic,f"coef.{int(ind_1460)}")
        np.save(wn_path, wn)
        np.save(coef_path, coef)

    # return nu4, coef4













