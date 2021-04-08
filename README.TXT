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

Download project's 3 files from MASTER repository and copy in your folder same az this:

your-folder__
             |__HMDB51__
             |          |__run
             |          |__sit
             |          |__...
             |
             |__estekhraj.py
             |__dataset_maker.py
             |__result.py
             
After run ( ESTEKHRAJ.PY ) you have this:

your-folder__
             |__opticalflow_proj_...__
             |                        |__doc__
             |__HMDB51__              |       |__...
             |          |__run        |
             |          |__sit        |__image__
             |          |__...        |         |__...
             |                        |
             |__estekhraj.py          |__kmeans__    
             |                        |          |__...
             |__dataset_maker.py      |         
             |                        |__periogram__
             |__result.py             |             |__...
                                      |             
                                      |__Image_error_list.txt
             
you need 2:20 second for make one periogram image and HMDB51 have 6849 videos. if you dont have time for making, you  will  download it from my google drive :              

             
             
