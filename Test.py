# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import brukerapi
from brukerapi.dataset import Dataset
from pathlib import Path
import os

# path to data cloned from
#data_path = Path('H:/MRI/AM7T_250121_SPC_extrudate_1_1_20250121_093706/1')
directory = 'H:/MRI/AM7T_250121_SPC_extrudate_1_1_20250121_093706/1/pdata/1/'
# both constructors are possible
dataset = Dataset('H:/MRI/AM7T_250121_SPC_extrudate_1_1_20250121_093706/1/pdata/1/2dseq/')
# dataset = Dataset(data_path / 'raw/Cyceron_DWI/20170719_075627_Lego_1_1/2')

print(dataset.data.shape)
# =============================================================================
# print(dataset.data.dtype)
# 
# print(dataset['VisuAcqSequenceName'].value)
# 
# print(dataset.id)
# print(dataset.affine)
# print(dataset.TE)
# print(dataset.TR)
# print(dataset.imaging_frequency)
# 
# print(dataset.affine)
# 
# print(dataset.dim_type)
# =============================================================================

