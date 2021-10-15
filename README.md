# mlip_workshop2021


## install

Needs `gcc > 5 or higher` and `cmake > 2.8 or higher`

```
conda create -n mlip_workshop python=3.9
conda install numpy scipy matplotlib sympy scikit-learn nglview tqdm
pip install skcosmo ase sympy spglib
pip install git+https://github.com/libAtoms/matscipy.git
pip install git+https://github.com/cosmo-epfl/librascal.git@workshop/mlip2021

```