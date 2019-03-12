#!/bin/bash

# NOTE: this script is useful for refreshing the code + data without rebuilding the container
#       copy and execute it in the parent directory containing the /rapids folder

# remove existing rapids directory, 
rm -rf rapids

# reclone from the latest repo
git clone https://github.com/miroenev/rapids

# rebuild data from kaggle survey for use in notebook plots
cd rapids 
mkdir kaggle_data 
mv *.zip kaggle_data 

cd kaggle_data 
unzip *.zip
wget -O results.csv https://raw.githubusercontent.com/adgirish/kaggleScape/d291e121b2ece69cac715b4c89f4f19b684d4d02/results/annotResults.csv
