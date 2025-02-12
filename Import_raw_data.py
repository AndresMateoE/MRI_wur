# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:39:59 2025

@author: mateo006
"""

import brukerapi
import subprocess

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
dir_experimet = '10'

''' Save folder '''
dir_save_folder = 'C:/Users/mateo006/Documents/Processed_data/250212_Test/'
#dir_save_name = ''

save_path = Path(dir_save_folder +  dir_experimet + '/')

save_path_folder = Path(dir_save_folder + dir_experimet + '/')
save_path_folder.mkdir(parents=True, exist_ok=True)

''' load data and make dataset'''
#print(dir_folder + '/' + dir_study + '/' + dir_experimet + '/pdata/1/')
raw_data_path = Path(dir_folder + '/' + dir_study + '/' + dir_experimet + '/')
print(raw_data_path)
print(save_path)
# =============================================================================
#                               Open matlab
'''
    Advice: Use this part to create the file but later comment it (Ctrl+4)
'''
# =============================================================================

matlab_exe = 'C:/MyPrograms/MatLab/bin/matlab.exe'

comando = [
    matlab_exe,
    '-batch',  # Esto ejecutará el script sin abrir la GUI de MATLAB
    f"load_rawdata_am('{raw_data_path}', '{raw_data_path}')"
]

subprocess.run(comando)





