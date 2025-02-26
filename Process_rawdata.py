# -*- coding: utf-8 -*-
"""
Read Raw Data and transform it to images

@author:   Andrés Mateo
"""

import numpy as np
import matplotlib.pyplot as plt
import Core_am as am
from pathlib import Path
from brukerapi.dataset import Dataset
from numpy.fft import fftshift, fft2
from scipy.optimize import curve_fit

#%% Load and Import rawdata
# =============================================================================
#                               Directory
# =============================================================================

''' load folder '''
dir_folder = 'C:/Users/mateo006/Documents/MRI/'
dir_study = 'AM7T_250225_SPI_SPIplus30'
dir_experimet = '11'

''' Save folder '''
dir_save_folder = 'C:/Users/mateo006/Documents/Processed_data/'.replace("\\", "/")
dir_save_name = '250226_SPI_SPI30Insect/'
save_path = Path(dir_save_folder + dir_save_name + '/' + dir_experimet)
save_path.mkdir(parents=True, exist_ok=True)

''' load data and make dataset'''
dataset = Dataset( dir_folder + '/' + dir_study + '/' + dir_experimet + '/pdata/1/')
dataset.add_parameter_file('method')  #add parameters PVM that sam uses.
sequence_name = dataset['VisuAcqSequenceName'].value
data2d = dataset.data


# =============================================================================
#                           Import rawdata 
#       (For now create it from the other script and the load it)
# =============================================================================

rawdata = np.genfromtxt(dir_folder + '/' + dir_study + '/' + dir_experimet + '/' + 'rawdata.csv', delimiter=',', dtype=str)
# Transform to the numpy complex format
rawdata = np.vectorize(am.convert_to_numpy_complex)(rawdata)


#Read other parameters according to the sequence
sequence_name = dataset['VisuAcqSequenceName'].value
print(sequence_name)

#%% RARE
if sequence_name == '<Bruker:RARE>':
    
    numRare = dataset['PVM_RareFactor'].value
    npoints = dataset._parameters['visu_pars']['VisuCoreSize'].value
    numSlices = dataset['PVM_SPackArrNSlices'].value
    numRepetitions = dataset['PVM_NRepetitions'].value
    numAverages = dataset['PVM_NAverages'].value
    numEchos = dataset['PVM_NEchoImages'].value
    slice_order = dataset['PVM_EncSteps1'].value
    image_order = dataset['PVM_ObjOrderList'].value
    
    
    #Create the new data format and reshape the data
    freqdata = np.zeros((len(rawdata[:,0]), len(rawdata[0,:]/(numRepetitions*numSlices)), numRepetitions, numSlices), dtype=complex)
    freqsize  = np.shape(freqdata)

    rawdata = np.reshape(rawdata, [freqsize[0],-1, numRepetitions])
    
    
    
    # Slice order
    slice_order = slice_order + (npoints[0]/2)
    slice_order = np.reshape(slice_order, [len(slice_order)//numRare,numRare]).astype(int)

# =============================================================================
#     print(slice_order)
#     print(npoints)
#     print(numRare)
#     print(numSlices)
#     print(numAverages)
#     print(numEchos)
#     print(dataset['PVM_EncSteps1'].value)
#     print(image_order)
# =============================================================================
    
    print(np.shape(slice_order))
    
    #Rawdata to k-space
    for rep in range(0,numRepetitions,1):
        #print('hi')
        for line in range(0,len(slice_order),1):
            #print(line)
            freqline = rawdata[:,line*numSlices*numRare:(line+1)*numSlices*numRare,rep]
            #print(freqline)
            for frame in range(0,numSlices):
                #print(frame)
                for rare in range(0,numRare):
                    freqdata[:,slice_order[line,rare],rep,image_order] = freqline[:,(frame*numRare + rare )]
                    
                    
    #imagedata = np.zeros(np.shape(freqdata))
    imagedata = np.zeros_like(freqdata, dtype=np.complex128)
    for slice in range (0,numSlices):
        imagedata[:,:,0,slice] = fftshift(fft2(freqdata[:,:,0,slice]))
        
    #Grafico k-space
# =============================================================================
#     plt.contour(np.abs(freqdata[:,:,0,0]), 
#                 levels = 2500,
#                 cmap = 'gray',
#                 lw=2)
#     plt.colorbar()
#     plt.show()
# =============================================================================
    im_data = np.squeeze(np.abs(imagedata[:,:,0,0]))
    im_data = im_data / (npoints[0]*npoints[1])
    
    #im_data = np.where(im_data > 3*np.mean(im_data), 3*np.mean(im_data), im_data)
    
    threshold = 0.8 * np.max(im_data)
    im_data = np.where(im_data > threshold, threshold, im_data)
    norm_im_data = im_data / threshold

    #rearrange data for correct ploting
    
    ROWS_TO_FLIP = -15
    COLUMS_TO_FLIP = 0
    
    plot_data = np.rot90(norm_im_data)
    plot_data = np.roll(plot_data, ROWS_TO_FLIP, 0)
    plot_data = np.roll(plot_data, COLUMS_TO_FLIP, 1)

    plt.figure(dpi=1200)    
    plt.imshow(plot_data,
               cmap='inferno',
               interpolation='none',
               
               )
    plt.colorbar()
    plt.savefig(save_path / "Process_fig_RARE", dpi=1200)
    plt.show()

#%% FLASH
elif sequence_name == '<Bruker:FLASH>':
    print('Use Load_2dsec for FLASH')

#%% MSME
elif sequence_name == '<Bruker:MSME>':
    

    # Load needed parameters
    PVM_Matrix = dataset['PVM_Matrix'].value
    nSlices = np.int64(dataset['PVM_SPackArrNSlices'].value)
    numEchos = dataset['PVM_NEchoImages'].value
    phaseOrder = np.array(dataset['PVM_EncSteps1'].value + PVM_Matrix[1]/2, dtype=int)
    sliceOrder = np.array(dataset['PVM_ObjOrderList'].value, dtype=int)

    
    # Arrange the rawdata file
    raw = np.reshape(rawdata, (PVM_Matrix[0], numEchos, nSlices, PVM_Matrix[1]))
    raw = np.transpose(raw, (0, 3, 2, 1))
    raw[:, phaseOrder, sliceOrder:sliceOrder+1, :] = raw
    raw = np.reshape(rawdata, (PVM_Matrix[0], PVM_Matrix[1], nSlices, numEchos))

    kspace = raw
    
# =============================================================================
#     plt.contour(np.abs(kspace[:,:,0,44]),levels = 2500,cmap = 'gray')
#     plt.colorbar()
#     plt.show()
# =============================================================================
    
    # Now process the kpace to image
    freqdata = kspace
    imagedata = np.zeros_like(freqdata, dtype=np.complex128)
    for slice in range(0,nSlices):
        for ee in range(0,numEchos):
            imagedata[:,:,slice,ee] = fftshift(fft2(freqdata[:,:,slice,ee]))
    
    
    #%% MSME Exponential fit
    fitdata = imagedata
    fitdata = np.abs(fitdata)
    fitdata = fitdata / np.max(fitdata[:,:,0,0])  #Normalize with first image
    TE = dataset.TE  #list with echo times
    
    #Define the function
    def T2_decay(t, M, T2, C):
        return M * np.exp(-t / T2) + C
        
    # 'Bulk' fit (like average)
    av_t_data = np.zeros(numEchos)
    for i in range(0,numEchos):
        temp_data = fitdata[:,:,0,i]
        temp_data = am.create_mask(temp_data, 6)
        temp_data = temp_data[temp_data>0]
        av_t_data[i] = np.mean(temp_data)
        
    T2_coef, T2_cov = curve_fit(T2_decay, TE, av_t_data, p0=[1, 12, 0.1], maxfev =5000)
    
    #print(T2_coef, T2_cov)
    T2_conf = np.array([T2_coef -  np.sqrt(np.diag(T2_cov)) * 1.96,  # Límite inferior
                    T2_coef + np.sqrt(np.diag(T2_cov)) * 1.96]) # Límite superior
    
    av_T2_error = T2_conf[1,1]-T2_conf[0,1]
    av_T2_value = T2_coef[1]
    
    print(av_T2_value, av_T2_error)
    
    # voxel by voxel fit
    sizedata = np.shape(imagedata)
    mapdata = np.zeros_like(fitdata[:,:,0,0])
    errordata = np.zeros_like(fitdata[:,:,0,0])
    
    for row in range(0,sizedata[0]):
        for col in range(0,sizedata[1]):
            temp_data = np.squeeze(fitdata[row,col,0,:])
            if temp_data[0] == 0:
                mapdata[row,col] = 0
                continue
            if np.any(np.isnan(temp_data)) or np.any(np.isinf(temp_data)):
                print("Error: temp_data contiene NaN o Inf")
            T2_coef, T2_cov = curve_fit(T2_decay, TE, temp_data, p0=[1, 12, 0.01], maxfev=5000)
            T2_conf = np.array([T2_coef -  np.sqrt(np.diag(T2_cov)) * 1.96,  # Límite inferior
                                T2_coef + np.sqrt(np.diag(T2_cov)) * 1.96]) # Límite superior
        
            T2_error = T2_conf[1,1]-T2_conf[0,1]
            T2_value = T2_coef[1]
            mapdata[row,col]= T2_value
            errordata[row,col] = T2_error
            
            #print(T2_value, T2_error)
            
    
    ROWS_TO_FLIP = 0
    COLUMS_TO_FLIP = 15
        
    #plot_data = np.rot90(norm_im_data)
    mapdata = np.roll(mapdata, ROWS_TO_FLIP, 0)
    mapdata = np.roll(mapdata, COLUMS_TO_FLIP, 1)

    mapdata = np.where(mapdata > 80, 0, mapdata)
    #mapdata, mask = am.create_mask3(mapdata, 7)    
    mapdata = am.create_mask(mapdata, 7)    

    plt.figure(dpi=1200)    
    img = plt.imshow(mapdata, 
               cmap='inferno',
               interpolation='none',
               
               )
    cbar = plt.colorbar(img)
    cbar.set_label("T2 [ms]", fontsize=12)
    #plt.savefig(save_path / "Process_fig_MSME", dpi=1200)
    plt.show()
            
    #%% MSME images for each echo
# =============================================================================
#     # Print an image for each echo, just to check 
#     for echo in range(0,numEchos):
#         
#         im_data = np.squeeze(np.abs(imagedata[:,:,0,echo]))
#         #im_data = im_data / (npoints[0]*npoints[1])
#     
#         threshold = 0.9 * np.max(im_data)
#         im_data = np.where(im_data > threshold, threshold, im_data)
#         norm_im_data = im_data / threshold
#     
#         #rearrange data for correct ploting
#         
#         ROWS_TO_FLIP = 25
#         COLUMS_TO_FLIP = 0
#         
#         plot_data = np.rot90(norm_im_data)
#         plot_data = np.roll(plot_data, ROWS_TO_FLIP, 0)
#         plot_data = np.roll(plot_data, COLUMS_TO_FLIP, 1)
#     
#         plt.figure(dpi=1200)    
#         plt.imshow(plot_data,
#                    cmap='inferno',
#                    interpolation='none',
#                    
#                    )
#         plt.colorbar()
#         plt.show()
#         pass
# =============================================================================

    
