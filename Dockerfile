# assumes root user [ nvidia-docker run -it --rm -u root ... ] 
# FROM nvcr.io/nvidia/rapidsai/rapidsai:cuda10.0_ubuntu16.04

FROM nvcr.io/nvidia/rapidsai/rapidsai:cuda10.0-runtime-ubuntu18.04

USER root 

RUN apt update && apt -y upgrade

RUN source activate rapids && conda install --yes -c conda-forge ipyvolume=0.5.1

RUN source activate rapids && jupyter nbextension enable --py --sys-prefix ipyvolume

RUN source activate rapids && jupyter nbextension enable --py --sys-prefix widgetsnbextension

RUN source activate rapids && conda install -c conda-forge ipywidgets

RUN source activate rapids && conda install -c conda-forge python-graphviz 

RUN apt -y install font-manager

RUN apt -y install unzip

RUN source activate rapids && pip install cupy-cuda100

RUN mkdir kaggle_data \
	&& cd kaggle_data \
	&& wget -O kaggle_survey.zip https://www.dropbox.com/s/0jk3v5in18atqom/kaggle-survey-2017.zip?dl=0 \
	&& unzip *.zip

RUN cd kaggle_data && wget -O results.csv https://raw.githubusercontent.com/adgirish/kaggleScape/d291e121b2ece69cac715b4c89f4f19b684d4d02/results/annotResults.csv

RUN wget -O rapids_demo_v5.ipynb https://www.dropbox.com/s/z5res5mp3qrp111/rapids_demo_v5.ipynb?dl=0
RUN wget -O fig_helpers.py https://www.dropbox.com/s/3c0ztgx6tn8scpp/fig_helpers.py?dl=1

EXPOSE 8888

CMD ["bash", "-c", "source activate rapids && jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=''"]
