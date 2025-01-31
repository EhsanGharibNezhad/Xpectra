
from setuptools import setup, find_packages
import re
# from codecs import open

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent

long_description = (this_directory / "README.md").read_text()
#
# with open(this_directory/"requirements.txt", "r") as fh:
#     install_requires = fh.readlines()

# Read the __version__.py file
with open('Xpectra/__version__.py', 'r') as f:
    vf = f.read()

# Obtain version from read-in __version__.py file
version = re.search(r"^_*version_* = ['\"]([^'\"]*)['\"]", vf, re.M).group(1)


setup(
    name='Xpectra',
    version = version,  # MAJOR.MINOR.PATCH
    description = 'An End-to-End Python Package for Analyzing Laboratory Spectra For Extrasolar Atmospheres, Generating Statistical Reports, and Visualizing Results',
    long_description = long_description,
    long_description_content_type='text/markdown',
    author = 'Ehsan (Sam) Gharib-Nezhad',
    author_email = 'e.gharibnezhad@gmail.com',
    url = 'https://ehsangharibnezhad.github.io/Xpectra',
    license = 'GPL-3.0',
    download_url = 'https://github.com/EhsanGharibNezhad/Xpectra',
    classifiers = [
                  'Intended Audience :: Science/Research',
                  'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                  'Operating System :: OS Independent' ,
                  'Programming Language :: Python',
                  'Programming Language :: Python :: 3',
                  'Topic :: Scientific/Engineering :: Astronomy',
                  'Topic :: Software Development :: Libraries :: Python Modules'
  ],
  packages=find_packages(exclude=('tests', 'docs')),
  install_requires=['numpy',
                    'bokeh',
                    'pandas',
                    'matplotlib',
                    'seaborn',
                    'sphinx==8.1.3',
                    'scipy',
                    'jupyterlab',
                    # 'scikit-learn==1.3.0',
                    ],
    zip_safe = False,
)
