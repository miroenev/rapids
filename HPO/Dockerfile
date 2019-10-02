FROM rapidsai/rapidsai:0.9-cuda10.0-runtime-ubuntu18.04

ENV CONDA_ENV rapids

RUN source activate $CONDA_ENV && \
    apt-get update && \
    apt-get install -y screen unzip git vim htop font-manager && \
    rm -rf /var/lib/apt/*

RUN source activate $CONDA_ENV && conda install -y -c conda-forge ipywidgets
RUN source activate $CONDA_ENV && jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN source activate $CONDA_ENV && conda install -y -c conda-forge ipyvolume
RUN source activate $CONDA_ENV && jupyter labextension install ipyvolume

RUN source activate $CONDA_ENV && conda install -c conda-forge python-graphviz

RUN source activate $CONDA_ENV && conda install -y -c rapidsai cupy

RUN source activate $CONDA_ENV && pip install git+https://github.com/dask/dask-kubernetes.git

RUN source activate $CONDA_ENV && mkdir /rapids-dask-hpo && cd /rapids-dask-hpo && git clone https://github.com/miroenev/rapids && cd rapids/HPO/ && python setup.py install

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
