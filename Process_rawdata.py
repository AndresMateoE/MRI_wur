# -*- coding: utf-8 -*-
"""
Read Raw Data and transform it to images

@author: mateo006
"""

import numpy as np
import matplotlib.pyplot as plt
import Core_am as am
from pathlib import Path
from brukerapi.dataset import Dataset
from numpy.fft import fftshift, fft2

# =============================================================================
#                               Directory
# =============================================================================

''' load folder '''
dir_folder = 'C:/Users/mateo006/Documents/MRI/'
dir_study = 'AM7T_250121_SPC_extrudate_1_1_20250121_093706'
dir_experimet = '10'

''' Save folder '''
dir_save_folder = 'C:/Users/mateo006/Documents/Processed_data/'.replace("\\", "/")
dir_save_name = '250212_Test/'
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

    threshold = 0.9 * np.max(im_data)
    im_data = np.where(im_data > threshold, threshold, im_data)
    norm_im_data = im_data / threshold

    #rearrange data for correct ploting
    
    ROWS_TO_FLIP = 25
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
    plt.savefig(save_path / "Test1_fig_RARE", dpi=1200)
    plt.show()

elif sequence_name == '<Bruker:FLASH>':
    print('Use Load_2dsec for FLASH')

elif sequence_name == '<Bruker:MSME>':
    #print(rawdata)
    # Load needed parameters
    PVM_Matrix = dataset['PVM_Matrix'].value
    nSlices = dataset['PVM_SPackArrNSlices'].value
    numEchos = dataset['PVM_NEchoImages'].value
    phaseOrder = np.array(dataset['PVM_EncSteps1'].value + PVM_Matrix[1]/2, dtype=int)
    sliceOrder = np.array(dataset['PVM_ObjOrderList'].value, dtype=int)
    print(sliceOrder)

    
    # Arrange the rawdata file
    raw = np.reshape(rawdata, (PVM_Matrix[0], numEchos, nSlices, PVM_Matrix[1]))
    
    raw = np.transpose(raw, (0, 3, 2, 1))
    print(np.shape(raw))
    
    #raw[:, phaseOrder, np.ix_(sliceOrder), :] = raw
    raw[:, phaseOrder, sliceOrder:sliceOrder+1, :] = raw
    
    kspace = raw
    
    plt.contour(np.abs(kspace[:,:,0,48]), 
                levels = 2500,
                cmap = 'gray',
                )
    plt.colorbar()
    plt.show()
    
    # Now process the kpace to image
    freqdata = kspace
    imagedata = np.zeros_like(freqdata, dtype=np.complex128)
    for slice in range(0,nSlices):
        for ee in range(0,numEchos):
            imagedata[:,:,slice,ee] = fftshift(fft2(freqdata[:,:,slice,ee]))
    
    im_data = np.squeeze(np.abs(imagedata[:,:,0,0]))
    #im_data = im_data / (npoints[0]*npoints[1])

    threshold = 0.9 * np.max(im_data)
    im_data = np.where(im_data > threshold, threshold, im_data)
    norm_im_data = im_data / threshold

    #rearrange data for correct ploting
    
    ROWS_TO_FLIP = 25
    COLUMS_TO_FLIP = 0
    
    plot_data = np.rot90(norm_im_data)
    plot_data = np.roll(plot_data, ROWS_TO_FLIP, 0)
    plot_data = np.roll(plot_data, COLUMS_TO_FLIP, 1)

# =============================================================================
#     plt.figure(dpi=1200)    
#     plt.imshow(plot_data,
#                cmap='inferno',
#                interpolation='none',
#                
#                )
#     plt.colorbar()
#     #plt.savefig(save_path / "Test1_fig_RARE", dpi=1200)
#     plt.show()
# =============================================================================



