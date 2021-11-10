# mlip workshop 2021

+ mlip_tutorial.ipynb is a functional draft of the hands-on part of the presentation I would like to make

+ tools is a folder containing a set of utilities to avoid clogging the notebook. It needs to be in the same folder as the notebook for the import to work properly.
## install

For the notebook to run smoothly one needs `gcc > 5 or higher`, `cmake > 2.8 or higher` and the following python packages

```
conda create -n mlip_workshop python=3.9
conda install numpy scipy matplotlib sympy scikit-learn nglview tqdm
pip install skcosmo ase sympy spglib
pip install git+https://github.com/libAtoms/matscipy.git
pip install git+https://github.com/cosmo-epfl/librascal.git@workshop/mlip2021

```