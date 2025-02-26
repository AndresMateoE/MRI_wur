# -*- coding: utf-8 -*-
"""
Core functions to use in other codes. Pun in here the analogue to MSME_RawToData and this things

@author: mateo006
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte

#%% General functions
# Function to convert Bruker complex format to numpy
def convert_to_numpy_complex(string):
    # Reemplazar 'i' por 'j' para hacer el formato compatible con numpy
    string = string.replace('i', 'j')
    return complex(string)

#Add function for MSME fitting

#%% Mask for images


def create_mask(image_matrix, mask_size):
    # Extraer las esquinas (background)
    background = np.concatenate([
        image_matrix[:mask_size, :mask_size],
        image_matrix[-mask_size:, :mask_size],
        image_matrix[:mask_size, -mask_size:],
        image_matrix[-mask_size:, -mask_size:]
    ], axis=0)
    

    # Calcular el promedio en las tres dimensiones (R, G, B o escala de grises)
    background_mean = np.mean(background)
    background_max = np.max(background)
    
    
    # Only first RGB channel
    image_gray = image_matrix[:, :, 0] if image_matrix.ndim == 3 else image_matrix
    
    # Normalize
    #image_gray = image_gray / np.max(image_gray)
    
    # gaussian filter
    smoothed = gaussian_filter(image_gray, sigma=2)
    
    # Binarización con umbral de Otsu
    thresh = threshold_otsu(smoothed)
    mask = smoothed > thresh
    
    # Restar el fondo
    image_matrix = image_matrix.astype(float)
    image_matrix[mask] -= background_mean
    image_matrix[image_matrix < 0] = 0
    
    # Expand to all RGB channels 
    if image_matrix.ndim == 3:
        mask = np.repeat(mask[:, :, np.newaxis], image_matrix.shape[2], axis=2)
    
    masked_image = np.copy(image_matrix)  # Copia para evitar modificar la original
    #masked_image[~mask] = 0  # Establecer en 0 donde la máscara es 0
    
    return masked_image

