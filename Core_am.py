# -*- coding: utf-8 -*-
"""
Core functions to use in other codes. Pun in here the analogue to MSME_RawToData and this things

@author: mateo006
"""

import numpy as np

# Function to convert Bruker complex format to numpy
def convert_to_numpy_complex(string):
    # Reemplazar 'i' por 'j' para hacer el formato compatible con numpy
    string = string.replace('i', 'j')
    return complex(string)

#Add function for MSME fitting