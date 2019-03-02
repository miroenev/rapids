FROM nvcr.io/nvidia/rapidsai/rapidsai:cuda10.0-runtime-ubuntu16.04

USER root 

RUN apt update && apt -y upgrade

RUN source activate rapids && conda install --yes -c conda-forge ipyvolume=0.5.1

RUN source activate rapids && jupyter nbextension enable --py --sys-prefix ipyvolume

RUN source activate rapids && jupyter nbextension enable --py --sys-prefix widgetsnbextension

RUN source activate rapids && conda install -c conda-forge ipywidgets

RUN source activate rapids && conda install -c conda-forge python-graphviz 

RUN apt -y --fix-missing install font-manager

RUN apt -y install unzip

RUN apt -y install git 

RUN git clone https://github.com/miroenev/rapids

RUN cd rapids && mkdir kaggle_data && mv *.zip kaggle_data && cd kaggle_data && unzip *.zip

RUN cd rapids && cd kaggle_data && wget -O results.csv https://raw.githubusercontent.com/adgirish/kaggleScape/d291e121b2ece69cac715b4c89f4f19b684d4d02/results/annotResults.csv

EXPOSE 8888

CMD ["bash", "-c", "source activate rapids && jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=''"]
