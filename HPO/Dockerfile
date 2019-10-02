FROM rapidsai/rapidsai:0.9-cuda10.0-runtime-ubuntu18.04

ENV CONDA_ENV rapids

RUN apt update && apt -y upgrade

RUN source activate $CONDA_ENV && conda install -y -c conda-forge ipywidgets
RUN source activate $CONDA_ENV && jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN source activate $CONDA_ENV && conda install -y -c conda-forge ipyvolume
RUN source activate $CONDA_ENV && jupyter labextension install ipyvolume

RUN source activate $CONDA_ENV && conda install -c conda-forge python-graphviz

RUN source activate $CONDA_ENV && conda install -y -c rapidsai cupy

RUN source activate $CONDA_ENV && pip install git+https://github.com/dask/dask-kubernetes.git

RUN source activate $CONDA_ENV && mkdir /rapids-dask-hpo && cd /rapids-dask-hpo && git clone https://github.com/miroenev/rapids && cd rapids/HPO/ && python setup.py install

EXPOSE 8888

WORKDIR /
