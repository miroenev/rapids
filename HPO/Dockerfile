# FROM rapidsai/rapidsai:0.8-cuda10.0-devel-ubuntu18.04-gcc7-py3.7
FROM nvcr.io/nvidia/rapidsai/rapidsai:0.9-cuda10.0-runtime-ubuntu18.04

RUN apt update && apt -y upgrade

RUN source activate rapids && conda install -y -c conda-forge nodejs
RUN source activate rapids && conda upgrade --all -y

RUN source activate rapids && conda install -y -c conda-forge ipywidgets
RUN source activate rapids && jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN source activate rapids && conda install -y -c conda-forge ipyvolume

RUN source activate rapids && jupyter labextension install ipyvolume
RUN source activate rapids && jupyter labextension install jupyter-threejs

RUN source activate rapids && conda install -c conda-forge python-graphviz 

RUN apt -y --fix-missing install unzip git vim htop

RUN git clone https://github.com/miroenev/rapids

# enables demo of ETL with RAPIDS and model building with DL-framework [ optional extension ]
RUN source activate rapids && conda install -y -c pytorch pytorch    

# 
RUN source activate rapids && conda install -y cupy
RUN source activate rapids && conda install -y -c rapidsai -c nvidia cuml=0.9
RUN source activate rapids && conda install -y -c rapidsai/label/xgboost xgboost=0.90.rapidsdev1

# jupyer lab
EXPOSE 8888
# dask dashboard
EXPOSE 8787

# activate jupyter lab on start
CMD ["bash", "-c", "source activate rapids && jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=''"]

# TODO fix fonts

