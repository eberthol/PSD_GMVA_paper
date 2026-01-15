# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:19:31 2022

@author: jianxin6

Emilie's modificaitons:
 - nofsamples=all_data[0][6] -> nofsamples=int(all_data[0][6]: nofsamples is now an integer and not an array of shape (1,)
 - sum -> np.sum (2 places) -> sum doesn't work properly on np arrays and was given wrong values (of the order 300 instead of order 1)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate
import math
#==
class Pulse_process_functions:
    
    def __init__(self, fileName):
        self.fileName = fileName
								   
        self.blockType = np.dtype([('Board',(np.int16,1)),
                                 ('Channel',(np.int16,1)),
                                 ('Time Stamp',(np.int64,1)),
                                 ('Energy',(np.int16,1)),
                                 ('Energy Short',(np.int16,1)),
                                 ('Flags',(np.int32,1)),
                                 ('Number of Wave samples to be read',(np.int32,1)),
                                 ('Samples',np.uint16,296)])
								   
        self.location = 0
    
    def GetNumberOfWavesInFile(self):
        return int(os.path.getsize(self.fileName) / self.blockType.itemsize)      
        
#===Function to get PSD spectrum of the accquired pulses and save tail,total,ttl to txt file==
    def Get_NG_ratio(self, numWaves, total_start_ch, tail_start_ch, case_nb):
        count = 0
        """Loads numWaves waveforms. If numWaves == -1, loads all waveforms in the file"""
        fid = open(self.fileName, "rb") ## rb for read binary
        fid.seek(self.location, os.SEEK_SET) ## tell computer where to start reading (i.e. at which byte)
        self.location += self.blockType.itemsize * numWaves
  # all_data contains all informations of pulses from the digitizer
        all_data=np.fromfile(fid, dtype = self.blockType, count=numWaves)         
  # digitizer settings:
        ## EMILIE
        nofsamples=int(all_data[0][6]) # number of samples per pulse (256)
        ## Least Significant Bit Size: conversion factor ADC <-> voltage
        ## 14-bit digitizer, 500 MSps
        ## signal sampled every 2 ns, input range of 0–2 V with a low-level threshold of 0.002 V
        v=2 #0.5V
        lsb=v/((2**14)-1) 

        nofpulses=numWaves 
  # PSD settings:
        tail_end=183  # the tail integration ends at 150 sample
        tail_start=tail_start_ch   # number of samples after the peak to start the tail integration
        total_start = total_start_ch
        file1 = open(f"Pulse_waveform_case{case_nb}"+".txt", 'w') 
        file2 = open(f"Pulse_label_case{case_nb}"+".txt", 'w') 
  # calculate pulse height and pulse intergal for each pulse and save it to txt file   
  # the saved txt file has "pulse height + tail integral + total integral + tail to total ratio"    
        for i in range(0, nofpulses):
            baseline_v = np.average(all_data[i][7][0:10]) ## compute the baseline of the pusle (average on the 10 first samples)
            ## max_v = max(baseline_v-all_data[i][7][0:nofsamples-1])*lsb ## not used
            max_ch = np.argmax(baseline_v-all_data[i][7][0:nofsamples-1]) ## get position of the peak
            total=baseline_v*(185)*lsb-np.sum(all_data[i][7][max_ch-total_start_ch:max_ch-total_start_ch+185])*lsb
            tail=baseline_v*(tail_end-tail_start)*lsb-np.sum(all_data[i][7][max_ch+tail_start:max_ch+tail_end])*lsb
            # ttl=tail/total

#            Stilbene_discrim_number=tail-(-0.0074114*total*total+0.2958*total+0.012972)
            total_cali=total*758.7
            if (tail>0 and tail<3 and total>0 and total_cali>80):
                for j in range (295):
                    file1.write(str("{:.4f}".format((baseline_v-all_data[i][7][j])*lsb))+" ")  
                file1.write(str("{:.4f}".format((baseline_v-all_data[i][7][295])*lsb))+"\n") 
                if (total<1.9):
                    Stilbene_discrim_number=tail-(-0.032605*total*total+0.32424*total+0.0024835)
                else:
                    Stilbene_discrim_number=tail-(0.22898*total+0.065817)
                if (Stilbene_discrim_number>0):
                    file2.write("1"+"\n") 
                    count+=1                     
                else:
                    file2.write("0"+"\n") 

        print(max_ch,total,tail)
        return count                   


## process all files
for case in range(1, 11):
    print(f'case{case}')
    mywave = Pulse_process_functions(fileName=f'binary_files/case{case}.bin')   
    nofwave=mywave.GetNumberOfWavesInFile()
    print(" Number of waves = "+str(nofwave)) 
    count = mywave.Get_NG_ratio(nofwave,2,9, case) #Get_NG_ratio(self, numWaves, total_start_ch, tail_start_ch, case_nb)
    print('neutrons=',count)

# # data=np.loadtxt('Pulse_waveform.txt')

# print('Done!')