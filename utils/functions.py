# Custom functions
import pandas as pd
import numpy as np
import os

from utils.config_reader import config_reader 
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, roc_auc_score
from zipfile import ZipFile




# # Импортируем константы из файла config
config = config_reader('../config/config.json')

def read_csv_from_zip(path:str=config.data_dir, exclusion:dict=None, index_col=None, ): #sep=None, filename=None, 
    
    mounts = dict()
    
    #lambda sep: ';' if sep is None else sep
    
    for archive in os.listdir(path):
        if archive in exclusion.keys():
            for filename, sep in exclusion.items():
                with ZipFile(os.path.join(config.data_dir, archive)) as myzip:
                    for file in myzip.namelist():
                        mounts[f"{filename[:-4]}".format(filename)] =  pd.read_csv(myzip.open(filename[:-4]+'.csv'), sep=sep)
        else:
            with ZipFile(os.path.join(config.data_dir, archive)) as myzip: 
                for file in myzip.namelist():
                    mounts[f"{file[:-4]}".format(file)] = pd.read_csv(myzip.open(file))             
        
                    
    return mounts

# def read_csv_from_zip1(path:str=config.data_dir, exclusion:list=None, index_col=None, ): #sep=None, filename=None, 
    
#     mounts = dict()
    
#     #lambda sep: ';' if sep is None else sep
    
#     for archive in os.listdir(path):
#         if archive not in exclusion:
#             with ZipFile(os.path.join(config.data_dir, archive)) as myzip: #as myzip
#                 for file in myzip.namelist():
#                     mounts[f"{file[:-4]}".format(file)] = pd.read_csv(myzip.open(file)) #, index_col=0 
#         # для файлов с сепаратором ';'            
#         else:
#             with ZipFile(os.path.join(config.data_dir, archive)) as myzip:
#                 for file in myzip.namelist():
#                     mounts[f"{file[:-4]}".format(file)] =  pd.read_csv(myzip.open(file), sep=';')
                    
#     return mounts