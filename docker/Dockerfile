FROM deepnote/python:3.8

RUN pip install numpy scipy matplotlib sympy scikit-learn nglview tqdm jupyter
RUN jupyter-nbextension enable nglview --py --sys-prefix
RUN pip install skcosmo ase sympy spglib pandas

RUN pip install git+https://github.com/libAtoms/matscipy.git

RUN pip install git+https://github.com/cosmo-epfl/librascal.git@workshop/mlip2021

# RUN git clone https://github.com/cosmo-epfl/librascal.git \
#  && cd librascal \
#  && git checkout workshop/mlip2021 \
#  && pip install .