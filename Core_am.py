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
from scipy.ndimage import gaussian_filter

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
    masked_image[~mask] = 0  # Establecer en 0 donde la máscara es 0
    

    
    return masked_image

def create_mask2(image_matrix, mask_size):
    border_size = mask_size  # Número de píxeles en los bordes

    # Extraer los bordes de la matriz
    edges = np.copy(image_matrix)
    edges[border_size:-border_size, border_size:-border_size] = 0  # Mantener solo los bordes

    # Aplicar un filtro (Gaussiano en este caso) a los bordes
    edges = np.clip(edges, 0, 100)
    filtered_edges = gaussian_filter(edges, sigma=2)

    # Insertar los bordes filtrados de nuevo en la matriz original
    filtered_matrix = np.copy(image_matrix)
    filtered_matrix[:border_size, :] = filtered_edges[:border_size, :]
    filtered_matrix[-border_size:, :] = filtered_edges[-border_size:, :]
    filtered_matrix[:, :border_size] = filtered_edges[:, :border_size]
    filtered_matrix[:, -border_size:] = filtered_edges[:, -border_size:]

    return filtered_matrix

def create_mask3(image_matrix, mask_size):
    """
    Aplica una máscara a la imagen eliminando el fondo.
    """
    
    # Extraer el fondo de las esquinas
    background = np.concatenate([
        image_matrix[:mask_size, :mask_size],  # Esquina superior izquierda
        image_matrix[-mask_size:, :mask_size],  # Esquina inferior izquierda
        image_matrix[:mask_size, -mask_size:],  # Esquina superior derecha
        image_matrix[-mask_size:, -mask_size:]  # Esquina inferior derecha
    ], axis=0)


    # Calcular el promedio del fondo
    background_mean = np.mean(background)
    
    # Restar el fondo y eliminar valores negativos
    image_matrix = np.maximum(image_matrix - background_mean, 0)
    
    # Generar la máscara: suavizado gaussiano y binarización
    mask = gaussian_filter(image_matrix / np.max(image_matrix), sigma=2)
    #_, mask = cv2.threshold(mask, 0.2, 1, cv2.THRESH_BINARY)  # Binarización
    #mask = mask.astype(np.uint8) * 255
    
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Cerrar agujeros en la máscara

    # Aplicar la máscara: eliminar el fondo
    masked_image = np.copy(image_matrix)
    masked_image[mask == 0] = 0
    
    return masked_image, mask

def create_mask4(image_matrix, mask_size):
    """
    Aplica una máscara a la imagen eliminando el fondo desde los bordes hacia adentro.
    """
    
    # --- 1. Extraer fondo desde los bordes ---
    border_top = image_matrix[:mask_size, :]
    border_bottom = image_matrix[-mask_size:, :]
    border_left = image_matrix[:, :mask_size]
    border_right = image_matrix[:, -mask_size:]

    background_pixels = np.concatenate([border_top, border_bottom, border_left, border_right], axis=None)
    

    # --- 2. Calcular el promedio del fondo ---
    background_mean = np.mean(background_pixels)
    
    # --- 3. Restar el fondo y eliminar valores negativos ---
    image_matrix = np.maximum(image_matrix - background_mean, 0)
    
    # --- 4. Crear máscara con desenfoque gaussiano ---
    mask = gaussian_filter(image_matrix / np.max(image_matrix), sigma=5)  # Sigma mayor para suavizar más
    _, mask = cv2.threshold(mask, 0.3, 1, cv2.THRESH_BINARY)  # Aumentamos el umbral para mantener más del objeto
    mask = mask.astype(np.uint8)  # Convertimos la máscara a entero (0 o 1)
    
    # --- 5. Filtrado morfológico para mejorar la máscara ---
    kernel = np.ones((5, 5), np.uint8)  # Kernel más grande para eliminar ruido externo
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Suaviza los bordes de la máscara
    
    # Convertir a booleano
    mask = mask.astype(bool)

    # --- 6. Aplicar máscara a la imagen ---
    masked_image = np.copy(image_matrix)
    masked_image[mask == 0] = 0  # Se eliminan los valores fuera del objeto
    
    return masked_image, mask
