# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 01:31:40 2024

@author: saura
"""

import os
import pyxdf
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time

chanNames = ['FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
             'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
             'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
             'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'F9', 'F10', 'P9', 'P10']
fmin =  4
fmax  = 40
sfreq = 250
eegA_name = 'NR-2022.09.05'
eegB_name = 'NR-2022.09.04'

# Define files:
baseFolder = 'F:\Saurabh_files\Downloads\eeg_hyperscan_for_saurabh\eeg_hyperscan_for_saurabh\sub-0431\ses-001\eeg-eeg-zephyr'
dataSave_filename =  'sub-0431_ses-001_task-stare_run-001_eeg-eeg-zephyr.xdf'
currFileName = os.path.join(baseFolder,dataSave_filename)

#%% Load the data:
tempdata, header = pyxdf.load_xdf(currFileName)

for tempIdx in range(len(tempdata)):
    currName = tempdata[tempIdx]['info']['name'][0]
    
    if eegA_name == currName: 
        eegA_data = tempdata[tempIdx]['time_series']
        eegA_time = tempdata[tempIdx]['time_stamps']
        
    if eegB_name == currName: 
        eegB_data = tempdata[tempIdx]['time_series']
        eegB_time = tempdata[tempIdx]['time_stamps']

numChans = eegA_data.shape[1]
maxTime = min(len(eegA_time),len(eegB_time))
combTime = eegA_time[:maxTime-1]
combData = np.concatenate((eegA_data.T[:,:maxTime-1], eegB_data.T[:,:maxTime-1]))
combChanNames = ['A_' + x for x in chanNames] + ['B_' + x for x in chanNames]

#%% Compute Inter-Brain Synchrony (IBS):

# indices = (np.arange(0,numChans),               # row indices
#            np.arange(0,numChans) + numChans)    # col indices

freqs = np.arange(fmin,fmax,0.2)
indices = ([0],     # row indices
           [1])     # col indices

conAll = []
for i in range(0,numChans):
    print('Processing Channel ' + str(i+1)  + '/' + str(numChans))
    con = spectral_connectivity_time(combData[np.newaxis,[i, i+numChans],:], freqs = freqs, method='plv',
                                   indices=indices, sfreq=sfreq, fmin=fmin, fmax=fmax)
    
    conAll.append(con.get_data())

#%% Plot IBS Matrix:
IBS = np.squeeze(np.array(conAll))

figure = plt.figure()
axes = figure.add_subplot(111)

caxes = axes.matshow(IBS)
figure.colorbar(caxes)
 
axes.set_xticklabels(['']+[str(x) for x in freqs])
axes.set_yticklabels(['']+chanNames)
 
plt.show()

#%% Create MNE raw structure:
info = mne.create_info(combChanNames, sfreq)
raw = mne.io.RawArray(combData, info)

# Clean channel names to be able to use a standard montage
new_names = dict(
    (ch_name, ch_name.rstrip(".").upper().replace("Z", "z").replace("FP", "Fp"))
    for ch_name in raw.ch_names
)
raw.rename_channels(new_names)

# 