FROM rapidsai/rapidsai:0.9-cuda10.0-runtime-ubuntu18.04

RUN apt update && apt -y upgrade

RUN source activate rapids && conda install -y -c conda-forge ipywidgets
RUN source activate rapids && jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN source activate rapids && conda install -y -c conda-forge ipyvolume
RUN source activate rapids && jupyter labextension install ipyvolume

RUN source activate rapids && conda install -c conda-forge python-graphviz

RUN source activate rapids && conda install -y -c rapidsai cupy

RUN source activate rapids && git clone https://github.com/miroenev/rapids && cd rapids/HPO/ && python setup.py install

EXPOSE 8888

WORKDIR /
