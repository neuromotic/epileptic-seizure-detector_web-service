#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:17:10 2019

@author: edwardmolina10
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy import signal
import ledapy
from IIR2Filter import IIR2Filter
from scipy.fftpack import fft
from scipy.signal import find_peaks
from joblib import load
from collections import Counter

"""   SIGNAL PREPROCESSING   """

eda_sampling_rate = 4
       
# BVP SCALING 
BVP_data = pd.read_csv('BVP.csv', header=None, float_precision='high', skiprows=2) #Skip timestamp and sampling frecuency rows
scaler_BVP = MinMaxScaler()
BVP_scaled = scaler_BVP.fit_transform(BVP_data)
np.savetxt('BVP_scaled.csv', BVP_scaled, delimiter=",", fmt='%.6f')

# EDA FILTERING
raw_eda = pd.read_csv('EDA.csv', header=None, float_precision='high', skiprows = 2)
raw_eda = np.ravel(raw_eda)
# Create an order 1 lowpass butterworth filter:
b, a = signal.butter(1, 1.5, btype='lowpass', fs = eda_sampling_rate)
# Use filtfilt to apply the filter:
eda_filtered = signal.filtfilt(b, a, raw_eda)
np.savetxt('EDA_filtered.csv', eda_filtered, delimiter=",", fmt='%.6f')

# EXTRACT PHASIC DATA (SCR) USING LEDAPY
SCR_data = ledapy.runner.getResult(eda_filtered, 'phasicdata', eda_sampling_rate, optimisation=0)
np.savetxt('SCR.csv', SCR_data, delimiter=",", fmt='%.6f')

#EDA_SCALING
scaler_EDA = MinMaxScaler()
EDA_scaled = scaler_EDA.fit_transform(eda_filtered.reshape(-1,1))
np.savetxt('EDA_filtered_scaled.csv', EDA_scaled, delimiter=",", fmt='%.6f')

# SCR SCALING
scaler_SCR = MinMaxScaler()
SCR_scaled = scaler_SCR.fit_transform(SCR_data.reshape(-1,1))
np.savetxt('SCR_scaled.csv', SCR_scaled, delimiter=",", fmt='%.6f')



"""   FEATURE EXTRACTION   """

eda_samples_per_window = 4 * 10 # sampling_frecuency * window_size
acc_samples_per_window = 32 * 10 # sampling_frecuency * window_size
bvp_samples_per_window = 64 * 10 # sampling_frecuency * window_size
eda_increment = eda_samples_per_window
acc_increment = acc_samples_per_window
bvp_increment = bvp_samples_per_window

EDA_data = pd.read_csv('EDA_filtered_scaled.csv', header=None, float_precision='high')            
SCR_data = pd.read_csv('SCR_scaled.csv', header=None, float_precision='high')            
ACC_data = pd.read_csv('ACC.csv', header=None)
BVP_data = pd.read_csv('BVP_scaled', header=None, float_precision='high')

eda_index_t = 0 # Top window index
eda_index_b = eda_samples_per_window - 1 # Bottom window index
acc_index_t = 0
acc_index_b = acc_samples_per_window - 1
bvp_index_t = 0
bvp_index_b = bvp_samples_per_window - 1

features = pd.DataFrame(columns = ['acc_std_x',
                                   'acc_std_y',
                                   'acc_std_z',
                                   'acc_net_std',
                                   'acc_var_filtered_x',
                                   'acc_var_filtered_y',
                                   'acc_var_filtered_z',
                                   'acc_diff_min_max_x',
                                   'acc_diff_min_max_y',
                                   'acc_diff_min_max_z',
                                   'acc_first_derivative_mean',
                                   'acc_first_derivative_std',
                                   'eda_std',
                                   'eda_fft_energy',
                                   'scr_integrated_amplitude',
                                   'scr_max_amplitude',
                                   'scr_number_peaks',
                                   'bvp_fft_energy'])
    
eda_data_length = len(EDA_data.index)
acc_data_length = len(ACC_data.index)
bvp_data_length = len(BVP_data.index)

while(eda_index_b < eda_data_length and acc_index_b < acc_data_length and bvp_index_b < bvp_data_length):
    
    EDA_window = EDA_data.loc[eda_index_t : eda_index_b]
    SCR_window = SCR_data.loc[eda_index_t : eda_index_b]
    ACC_window = ACC_data.loc[acc_index_t : acc_index_b]
    BVP_window = BVP_data.loc[bvp_index_t : bvp_index_b]
    
    # ACC FEATURES
    
    # Standard deviation
    acc_std = np.std(ACC_window)
    acc_std_x = acc_std[0] #f1
    acc_std_y = acc_std[1] #f2
    acc_std_z = acc_std[2] #f3
    
    # Net-ACC STD
    acc_net = np.sqrt(np.power(ACC_window[[0]],2).to_numpy() + np.power(ACC_window[[1]],2).to_numpy() + np.power(ACC_window[[2]],2).to_numpy())
    acc_net_std = np.std(acc_net) #f4
    
    # Variance low-pass filter
    Filter_ACC = IIR2Filter(1,[0.5],'lowpass',fs=32)
    ACC_window_size = len(ACC_window.index)
    acc_x_filtered = np.zeros(ACC_window_size)
    acc_y_filtered = np.zeros(ACC_window_size)
    acc_z_filtered = np.zeros(ACC_window_size)                
    for i in range(ACC_window_size):
        acc_x_filtered[i] = Filter_ACC.filter(ACC_window.iloc[i, 0])
        acc_y_filtered[i] = Filter_ACC.filter(ACC_window.iloc[i, 1])
        acc_z_filtered[i] = Filter_ACC.filter(ACC_window.iloc[i, 2])
    acc_var_filtered_x = float(np.var(acc_x_filtered)) # f5
    acc_var_filtered_y = float(np.var(acc_y_filtered)) # f6
    acc_var_filtered_z = float(np.var(acc_z_filtered)) # f7
    
    # Difference min/max low pass filter
    acc_diff_min_max_x = np.amax(acc_x_filtered) - np.amin(acc_x_filtered) #f8
    acc_diff_min_max_y = np.amax(acc_y_filtered) - np.amin(acc_y_filtered) #f9
    acc_diff_min_max_z = np.amax(acc_z_filtered) - np.amin(acc_z_filtered) #f10
    
    # Accumulate first derivative mean
    acc_first_derivative_accumulate = np.sqrt(np.power(np.gradient(acc_x_filtered),2) + np.power(np.gradient(acc_y_filtered),2) + np.power(np.gradient(acc_z_filtered),2))
    acc_first_derivative_mean = np.mean(acc_first_derivative_accumulate) #f11
    
    # Accumulate first derivative STD
    acc_first_derivative_std = np.std(acc_first_derivative_accumulate) # f12
    
    # EDA FEATURES
    
    # Standard deviation
    eda_std = np.std(EDA_window) # f13
    
    # FFT energy
    eda_window_len = len(EDA_window)
    eda_fft = fft(EDA_window)/eda_window_len
    eda_fft_abs = abs(eda_fft)
    
    cuad=[0]*len(eda_fft_abs)
    for i in range(0,len(eda_fft_abs)):
        cuad[i]=eda_fft_abs[i]*eda_fft_abs[i]
    
    energy=0
    for i in cuad:
        energy=energy+i

    eda_fft_energy = energy # f14        
    
    # SCR FEATURES
    
    # Integrated amplitude
    scr_integrated_amplitude = np.sum(SCR_window) #f15
    
    # Max. amplitude
    scr_max_amplitude = np.amax(SCR_window) #f16
    
    # Number of peaks
    scr_number_peaks = len(find_peaks(SCR_window.to_numpy().ravel())[0]) # f17
    
    # BVP FEATURES
    
    # FFT energy
    bvp_window_len = len(BVP_window)
    bvp_fft = fft(BVP_window)/bvp_window_len
    bvp_fft_abs = abs(bvp_fft)
    bvp_fft_energy = energy(bvp_fft_abs) # f18
    
    features = features.append({'acc_std_x': float(acc_std_x),
                                'acc_std_y': float(acc_std_y),
                                'acc_std_z': float(acc_std_z),
                                'acc_net_std': float(acc_net_std),
                                'acc_var_filtered_x': float(acc_var_filtered_x),
                                'acc_var_filtered_y': float(acc_var_filtered_y),
                                'acc_var_filtered_z': float(acc_var_filtered_z),
                                'acc_diff_min_max_x': float(acc_diff_min_max_x),
                                'acc_diff_min_max_y': float(acc_diff_min_max_y),
                                'acc_diff_min_max_z': float(acc_diff_min_max_z),
                                'acc_first_derivative_mean': float(acc_first_derivative_mean),
                                'acc_first_derivative_std': float(acc_first_derivative_std),
                                'eda_std': float(eda_std),
                                'eda_fft_energy': float(eda_fft_energy),
                                'scr_integrated_amplitude': float(scr_integrated_amplitude),
                                'scr_max_amplitude': float(scr_max_amplitude),
                                'scr_number_peaks': float(scr_number_peaks),
                                'bvp_fft_energy': float(bvp_fft_energy)},
                                ignore_index=True)
    
    eda_index_t = eda_index_t + eda_increment
    eda_index_b = eda_index_b + eda_increment
    acc_index_t = acc_index_t + acc_increment
    acc_index_b = acc_index_b + acc_increment
    bvp_index_t = bvp_index_t + bvp_increment
    bvp_index_b = bvp_index_b + bvp_increment



"""   CLASSIFICATION   """

scaler_production = load('scaling.joblib')
features = scaler_production.transform(features)

classifier_production = load('RandomForestClassifier.joblib')
prediction = classifier_production.predict(features)

count = Counter(prediction)
print(count[1])