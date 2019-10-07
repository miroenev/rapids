FROM rapidsai/rapidsai:0.9-cuda10.0-runtime-ubuntu18.04

ENV CONDA_ENV rapids

RUN source activate $CONDA_ENV && \
    apt-get update && \
    apt-get install -y screen unzip git vim htop font-manager && \
    rm -rf /var/lib/apt/*

RUN source activate $CONDA_ENV && conda install -y -c conda-forge ipywidgets && conda clean -yac *
RUN source activate $CONDA_ENV && jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN source activate $CONDA_ENV && conda install -y -c conda-forge ipyvolume && conda clean -yac *
RUN source activate $CONDA_ENV && jupyter labextension install ipyvolume

RUN source activate $CONDA_ENV && conda install -c conda-forge python-graphviz && conda clean -yac *

RUN source activate $CONDA_ENV && conda install -y -c rapidsai cupy && conda clean -yac *

# There is currently a PR in for dask_kubernetes that adds a lot of new functionality, we are installing directly from there for now
RUN source activate $CONDA_ENV && pip install git+https://github.com/dask/dask-kubernetes.git
#RUN source activate $CONDA_ENV && pip install git+https://github.com/dask/distributed.git@master
#RUN source activate $CONDA_ENV && pip install git+https://github.com/jacobtomlinson/dask-kubernetes.git@spec-cluster

RUN source activate $CONDA_ENV && mkdir /rapids-dask-hpo && cd /rapids-dask-hpo && git clone https://github.com/miroenev/rapids && cd rapids/HPO/ && git pull && python setup.py install
RUN source activate $CONDA_ENV && mkdir /k8s-rapids-dask && cd /k8s-rapids-dask && git clone --depth=1 https://github.com/supertetelman/k8s-rapids-dask

# There is currently a bug in the jupyterlab-manager that requires install directly from Git
# When this bug is patched we should remove this piece.
# I have left the above `jupyter labextension install @jupyter-widgets/jupyterlab-manager && \` for simplicity, but it is an extra step
RUN source activate $CONDA_ENV && \
    cd /tmp && \
    git clone https://github.com/rapidsai/jupyterlab-nvdashboard.git && \
    cd jupyterlab-nvdashboard && \
    pip install -e . && \
    jlpm install && \
    jlpm run build && \
    jupyter labextension install . 

EXPOSE 8888

WORKDIR /
