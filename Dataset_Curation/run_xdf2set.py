# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:51:50 2020

@author: shaw5
"""
import xdf2set

filename = "P1001_pretrain2.xdf"
file_location = "Z:\Research_data\20191211_DatabaseBCI_EEG-P1001_Pretrain"
electrode_range = range(1,128)
electrode_file = "Y:\expts\AttentionBCI\Cap_files\Biosemi128New_NZ_LPA_RPA.sfp"
dataset_name = "P1001_pretrain2"

xdf2set(filename,electrode_range,electrode_file,dataset_name,file_location);