FROM rapidsai/rapidsai:0.11-cuda10.1-runtime-ubuntu18.04

ENV CONDA_ENV rapids

RUN source activate $CONDA_ENV && \
    apt-get update && \
    apt-get install -y screen unzip git vim htop font-manager && \
    rm -rf /var/lib/apt/*

RUN source activate $CONDA_ENV && jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN source activate $CONDA_ENV && conda install -y -c conda-forge ipyvolume && conda clean -yac *
RUN source activate $CONDA_ENV && jupyter labextension install ipyvolume

RUN source activate $CONDA_ENV && conda install -c conda-forge python-graphviz && conda clean -yac *

RUN source activate $CONDA_ENV && pip install dask-kubernetes

# clone latest code from github, and package HPO library files for notebook [ swarm.py, visualization.py, data_utils.py]
# note: this is necessary to enable dask-workers on Kubernetes Clusters to import HPO library
RUN source activate $CONDA_ENV && mkdir /rapids-dask-hpo && cd /rapids-dask-hpo \
    && git clone https://github.com/miroenev/rapids \
    && cd rapids/HPO && python setup.py install

EXPOSE 8888
EXPOSE 8787

WORKDIR /
