FROM deepnote/python:3.8

RUN pip install numpy scipy matplotlib sympy scikit-learn nglview tqdm jupyter 
RUN jupyter-nbextension enable nglview --py --sys-prefix
RUN pip install skcosmo ase sympy spglib

RUN pip install git+https://github.com/libAtoms/matscipy.git

RUN pip install git+https://github.com/cosmo-epfl/librascal.git@workshop/mlip2021
