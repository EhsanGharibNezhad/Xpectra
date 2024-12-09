.. TelescopeML documentation master file, created by
   sphinx-quickstart on Tue Dec 27 15:39:09 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Xpectra
==============


For analyzing exoplanetary spectra in the lab, providing insights into atmospheric composition and habitability. The ``Xpectra`` workflow is as follows:

.. image:: figures/xpectra_flowchart_1.jpg
   :width: 600

.. image:: figures/xpectra_flowchart_2.jpg
   :width: 600

``Xpectra`` is a Python package comprising a series of modules, each equipped with curve fitting and statistical capabilities for conducting spectral analysis tasks on molecular and atomic spectra recorded from laboratory spectroscopic measurements to understand the atmospheres of extrasolar planets and brown dwarfs.
The tasks executed by the ``Xpectra`` modules are outlined below:


- *SpecFitAnalyzer module*: Implements curve fitting to subtract baseline and predict spectroscopic parameters:

  - Preprocess laboratory spectrum
  - Correct spectral baseline 
  - Fit spectral peaks
  - Extract spectroscopic parameters 

- *SpecStatVisualizer module*: Utilizes interactive plotting with Bokeh to explore the data:

  - Visualizing the data in specified range
  - Explore spectral features 
  - Represent results 

- *LineAssigner module*: Parses HITRAN line lists to load and identify spectral lines:

  - Load and parse HITRAN line list
  - Tabulate spectroscopic parameters 
  - Identify spectral lines 

- *FitLiteratureData module*: Implements parallel processing to update HITRAN line list using literature:

  - Collect and vet literature coefficients 
  - Fit pressure-broadening accross quantum numbers, symmetry, and bands
  - Modify and update HITRAN line list 


or simply...

 - Load laboratory spectra
 - Follow the tutorials
 - Label the quantum assignments by connecting to the HITRAN database
 - Extract spectroscopic parameters, e.g., line position, pressure-broadening coefficients
 - Report the statistical analysis




======================

.. toctree::
   :maxdepth: 2
   :hidden:
  
   Installation <installation>
   Tutorials <tutorials>
   The Code <code>
   GitHub <https://github.com/EhsanGharibNezhad/Xpectra>

.. KnowledgeBase <knowledgebase>
.. Publications <publications>
.. What to Cite <cite>


