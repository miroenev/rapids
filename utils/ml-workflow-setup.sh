#!/env/sh
# This is a simple script meant to help install the dependencies required to run this demo in an existing RAPIDS container environment.
# Execute this script on in your running docker container and the notebooks should run properly.

notebook_dir=$1
repo_dir=ml_workflow

if [ -z ${notebook_dir} ]; then
    notebook_dir="/rapids/notebooks"
fi

apt-get update
apt-get install -y \
    unzip \
    font-manager \
    python-graphviz

pip install \
    matplotlib \
    ipyvolume

cd ${notebook_dir}
git clone https://github.com/miroenev/rapids.git ${repo_dir}
cd ${repo_dir}

mkdir kaggle_data
wget -O results.csv https://raw.githubusercontent.com/adgirish/kaggleScape/d291e121b2ece69cac715b4c89f4f19b684d4d02/results/annotResults.csv

mv *csv kaggle_data
mv *zip kaggle_data
unzip kaggle_data/*zip -d kaggle_data

echo "Down installing dependencies and downloading data. See ${notebook_dir}/${repo_dir}"
