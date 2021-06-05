## Setup 

wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
sh Anaconda3-2021.05-Linux-x86_64.sh
conda create -n py37 python=3.7 anaconda
conda activate py37
sudo apt install libfreetype-dev
sudo apt-get install libfontconfig1-dev
sudo apt-get install libopenblas-dev
sudo apt-get install libhdf5-dev
sudo apt install python3-pip
pip install -r requirements.txt
pip install tensorflow==1.13.2

## Create dataset with MaxQuant Score instead of CCS
python process_data_final.py evidence.txt

## Profit!
python run_training.py


# CCS Model Training and Prediction

Publication:
- doi: https://doi.org/10.1101/2020.05.19.102285
- biorxiv: https://www.biorxiv.org/content/10.1101/2020.05.19.102285v1

## Library Setup

Setup CUDA 10.0 with cudnn and install the required python libraries with pip:

```
pip install -r requirements.txt
```

## Prediction with Pre-Trained Model

Unzip the checkpoint found in out.

Prepare a csv file that contains Sequence and Charge Information and use the provided `predict.py` script:
```
python predict.py <filename.csv> 
```
For the format see the provided example file in `./data/combined_reduced.csv`

## Process data
Use the provided notebook: `process_data_final.ipynb`

It uses the raw data files and saves train and test files in pkl format to disc in `./data_final`

## Training

The `bidirectional_lstm.py` file contains training and prediction routines.

Training is done by setting the paths in `run_training.py` and executing it.
The complete dataset will be uploaded at a later stage of publication.

## Evaluation

Use the provided `evaluate.ipynb` Jupyter Notebook.

