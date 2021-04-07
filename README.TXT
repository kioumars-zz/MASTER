# MASTER

This codes run easyest in linux

Make your folder and Execute commands from the command line inside the folder

You must update ubuntu:

sudo apt update
sudo apt upgrade
audo apt autoremove

Download and Install anaconda latest version from https://www.anaconda.com/products/individual#linux 

Download and extract HMDB51 dataset from https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads

For Install OPENCV2 use this:

sudo apt-get install python3-opencv
OR
pip3 install opencv-python

You must install Other libraries from CONDA :

conda install -c anaconda keras
conda install -c anaconda pandas
conda install -c anaconda seaborn
conda install -c anaconda tensorflow
conda install -c anaconda matplotlib
conda install -c anaconda scikit-learn
and ...


Download project's 3 files from MASTER repository and copy in your folder

your-folder__
             |__HMDB51__
             |          |__run
             |          |__sit
             |          |__...
             |
             |__estekhraj.py
             |__dataset_maker.py
             |__result.py
             
After run you have this:

your-folder__
             |__HMDB51__
             |          |__run
             |          |__sit
             |          |__...
             |
             |__doc...__
             |          |__...
             |      
             |__image...__
             |             |__...
             |
             |__kmeans...__
             |             |__...
             |
             |__periogram...__
             |                |__...
             |
             |__Image_error_list.txt
             |__estekhraj.py
             |__dataset_maker.py
             |__result.py

             
             
