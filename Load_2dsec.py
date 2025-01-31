# -*- coding: utf-8 -*-
"""
Import and plot 2dseq bruker files
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
dir_folder = 'C:/Users/mateo006/Documents/MRI/'
dir_study = 'AM7T_250121_SPC_extrudate_1_1_20250121_093706'
dir_experimet = '1'

''' Save folder '''
dir_save_folder = 'C:/Users/mateo006/Documents/Processed_data/'.replace("\\", "/")
dir_save_name = '250131_Test/'
save_path = Path(dir_save_folder + dir_save_name + '/' + dir_experimet)
save_path.mkdir(parents=True, exist_ok=True)

''' load data and make dataset'''
dataset = Dataset( dir_folder + '/' + dir_study + '/' + dir_experimet + '/pdata/1/')
sequence_name = dataset['VisuAcqSequenceName'].value
data2d = dataset.data
#print(data2d)

# =============================================================================
#                           Prepearing the data
# =============================================================================

print(dataset.TE)
print(dataset.TR)
print(dataset.imaging_frequency)
print(dataset.affine)
print(dataset['VisuAcqSequenceName'])
print(dataset._parameters['visu_pars']['VisuCoreExtent'].value)

''' Reading FOV '''
fov = dataset._parameters['visu_pars']['VisuCoreExtent'].value
print(fov)
npoints = dataset._parameters['visu_pars']['VisuCoreSize'].value
print(npoints)
resolution = fov[0]/npoints[0]
print(resolution)
''' with this we can make the axix '''
axis_x = np.round(resolution * np.linspace(0, npoints[0], npoints[0]), 4)
axis_y = np.round(resolution * np.linspace(0, npoints[1], npoints[1]), 4)
#print(axis_x)
#print(axis_y)


# =============================================================================
#                                Ploting 
# =============================================================================


for i in range(len(dataset.data[0,0,:])):
    
    #plt.figure(figsize=(base_fig_size, fig_ratio * base_fig_size))
    plt.contour(
                #axis_x,
                #axis_y,
                dataset.data[:,:,i], 
                levels = 300, 
                cmap='magma'
                )
    plt.title(sequence_name)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.xticks(axis_x)
    #plt.yticks(axis_y)
    #plt.xlim(0, npoints[0])
    #plt.ylim(0, npoints[1])
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.colorbar()
    plt.savefig(save_path / f"Test1_fig_{i+1}", dpi=600)
    plt.show()
    

# =============================================================================
#                           Usefull lines
# =============================================================================

#print(dataset._parameters['visu_pars']['VisuCoreExtent'])  # Para ver qué tipo de objeto es
#print(dir(dataset))   # To see atributes and methods


