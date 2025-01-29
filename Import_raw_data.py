# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:39:59 2025

@author: mateo006
"""

import brukerapi
from brukerapi.dataset import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
#                              Path selection
# =============================================================================
''' load folder '''
dir_folder = 'C:/Users/mateo006/Documents/MRI'
dir_study = 'AM7T_250121_SPC_extrudate_1_1_20250121_093706'
dir_experimet = '4'

''' Save folder '''
dir_save_folder = 'C:/Users/mateo006/Documents/Processed_data/29012025_Test/'.replace("\\", "/")
dir_save_name = 'Test'
save_path = Path(dir_save_folder + dir_save_name + '/' + dir_experimet)
save_path.mkdir(parents=True, exist_ok=True)

''' load data and make dataset'''
#print(dir_folder + '/' + dir_study + '/' + dir_experimet + '/pdata/1/')
raw_data_path = dir_folder + '/' + dir_study + '/' + dir_experimet + '/pdata/1/'
print(raw_data_path)
dataset = Dataset(raw_data_path, state={ 'mmap': False, 'type': 'rawdata'})
dataset._set_state({'rawdata'})
print(dataset.type)
#sequence_name = dataset['VisuAcqSequenceName'].value

#dataset.__getstate__()

#help(Dataset)

#data2d = dataset.data

#print(data2d)
#acqp = dataset._parameters['visu_pars']
#print(acqp)

#print(data2d)

#print(dir(dataset))
#dir(dataset)
#dataset['schema']