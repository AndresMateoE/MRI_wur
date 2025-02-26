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
dir_study = 'AM7T_250225_SPI_SPIplus30'
dir_experimet = '12'


''' load data and make dataset'''

raw_data_path = Path(dir_folder + '/' + dir_study + '/' + dir_experimet + '/')

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





