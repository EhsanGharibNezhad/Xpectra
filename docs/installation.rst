Installation
=============

.. toctree::
   :maxdepth: 1
   :caption: Contents:



.. note::
    `Xpectra_project` requires python >= 3.11.


Step 1: Create your directory structure
----------------------------------------
Let’s start by creating the folder structure as follow, named *Xpectra_project*. While you are inside this parent *Xpectra_project* directory, download the *reference_data* folder which include the following sub-directories: *datasets*, *processed_data*, *tutorials*,  *figures*.

Download link for **reference_data** folder is: `Link <link>`_

| Xpectra_project
| ├── reference_data
| │   ├── datasets
| │   ├── processed_data
| │   ├── tutorials
| │   └── figures
|




Step 2: Set input file environment variables
---------------------------------------------


For Mac OS
++++++++++++++++++++++++++++++++++++++++++++

Follow the following steps to set the link to the input data:

1. Check your default shell in your terminal:

.. code-block:: bash

    echo $SHELL

This command will display the path to your default shell, typically something
like `/bin/bash` or `/bin/zsh`, or `/bin/sh`.

2. Set the environment variables :

    * If your shell is `/bin/zsh`:

    .. code-block:: bash

        echo 'export Xpectra_reference_data="/PATH_TO_YOUR_reference_data/" ' >>~/.zshrc
        source ~/.zshrc
        echo $Xpectra_reference_data


    * if your shell is `/bin/bash`:

    .. code-block:: bash

        echo 'export Xpectra_reference_data="/PATH_TO_YOUR_reference_data/"' >>~/.bash_profile
        source ~/.bash_profile
        echo $Xpectra_reference_data

    * if your sell is `/bin/sh`:

    .. code-block:: bash

        echo 'export Xpectra_reference_data="/PATH_TO_YOUR_reference_data/"' >>~/.profile
        source ~/.profile
        echo $Xpectra_reference_data


.. note::
    - Replace `PATH_TO_YOUR_reference_data` with the actual path to your *reference_data* folder
      that you downloaded in step 1.
    - *echo* command is used to check that your variable has been defined properly.


For Linux
++++++++++
In Linux, the choice between `~/.bashrc` and `~/.bash_profile` depends on your specific use case and how you want environment variables to be set, but `~/.bashrc` is a common and practical choice for modern Linux system.

.. code-block:: bash

    echo 'export Xpectra_reference_data="/PATH_TO_YOUR_reference_data/" ' >>~/.bashrc
    source ~/.bashrc
    echo $Xpectra_reference_data



Step 3: Install the Package
----------------------------

.. note::
    You need to first have `Anaconda distribution <https://www.anaconda.com/download/>`_ installed on your machine, before proceed to the next steps.


If you want to access the latest features or modify the code and contribute, we suggest that you clone the source code from GitHub by following steps:

.. note::
    For best practise, it is recommended to be inside the `Xpectra_project` parent directory and then clone the github repository.

1. Clone the repo and Create `Conda` environment named *Xpectra*:

.. code-block:: bash

    git clone https://github.com/ehsangharibnezhad/Xpectra.git
    cd Xpectra
    conda env create -f environment.yml



2. Activate the new environment:

.. code-block:: bash

    conda activate Xpectra


3. Install the `Xpectra` Library

You can install the `Xpectra` library using one of the following methods:

3.1 For Development Purposes:

If you plan to develop the code, navigate to the *Xpectra* directory and install it using:

.. code-block:: bash

    python3.11 setup.py develop


3.2 For General Use:

If you intend to use the code without making changes in the future, install the PyPI version:

.. code-block:: bash

    pip install Xpectra

4. Test the package by going to the **docs/tutorials/** directory and run all notebooks there using *jupyter-lab*.
