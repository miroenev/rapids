# FROM nvcr.io/nvidia/rapidsai/rapidsai:0.7-cuda10.0-runtime-ubuntu18.04-gcc7-py3.7
FROM rapidsai/rapidsai:0.11-cuda10.1-runtime-ubuntu18.04

ENV CONDA_ENV rapids

RUN apt update && apt -y upgrade

RUN source activate $CONDA_ENV && conda install -y -c conda-forge nodejs
RUN source activate $CONDA_ENV && conda install -y -c conda-forge ipywidgets
RUN source activate $CONDA_ENV && jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN source activate $CONDA_ENV && conda install -y -c conda-forge ipyvolume

RUN source activate $CONDA_ENV && jupyter labextension install ipyvolume
RUN source activate $CONDA_ENV && jupyter labextension install jupyter-threejs

RUN source activate $CONDA_ENV && conda install -c conda-forge python-graphviz 

RUN apt -y --fix-missing install font-manager unzip git vim htop

RUN git clone https://github.com/miroenev/rapids

# enables demo of ETL with RAPIDS and model building with DL-framework [ optional extension ]
RUN source activate $CONDA_ENV && conda install -y -c pytorch pytorch    

EXPOSE 8888

CMD ["bash", "-c", "source activate $CONDA_ENV && jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=''"]
