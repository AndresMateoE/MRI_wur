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

# =============================================================================
#                               Directory
# =============================================================================

''' load folder '''
dir_folder = 'C:/Users/mateo006/Documents/MRI/'
dir_study = 'AM7T_250121_SPC_extrudate_1_1_20250121_093706'
dir_experimet = '8'

''' Save folder '''
dir_save_folder = 'C:/Users/mateo006/Documents/Processed_data/'.replace("\\", "/")
dir_save_name = '250203_Test/'
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

rawdata = np.genfromtxt(dir_save_folder + dir_save_name + '/' + dir_experimet + '/' + 'rawdata.csv', delimiter=',', dtype=str)
# Transform to the numpy complex format
rawdata = np.vectorize(am.convert_to_numpy_complex)(rawdata)
print(rawdata.shape)

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
    print(freqsize[0])
    print(numRepetitions)
    rawdata = np.reshape(rawdata, [freqsize[0],-1, numRepetitions])
    
    
    # Slice order
    slice_order = slice_order + (npoints[0]/2)
    slice_order = np.reshape(slice_order, [len(slice_order)//numRare,numRare]).astype(int)
    #print(slice_order)
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
    

    
    #Rawdata to k-space
    for rep in range(0,numRepetitions,1):
        #print('hi')
        for line in range(0,len(slice_order),1):
            #print(line)
            freqline = rawdata[:,line*numSlices*numRare:(line+1)*numSlices*numRare,rep]
            #print(freqline)
            for frame in range(0,numSlices):
                print(frame)
                for rare in range(0,numRare):
                    print(rare)
                    #print('.')
                    #freqdata[:,slice_order[rare,line],rep,image_order] = freqline[:,(frame*numRare + rare )]









