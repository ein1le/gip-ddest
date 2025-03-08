#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 18:34:48 2025

@author: demetrigaffney
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Fundamental constants
e = np.e
π = np.pi

# Number of experiments 
n = 5

'''
IMPORT TESTING DATA
-------------------
'''

# Import csv data
data1 = []
data2 = []

for i in range(1, n):
    csv_name1 = f"Test{i}.steps.csv"
    csv1 = pd.read_csv(csv_name1, header=None, skiprows=1)
    data1.append(csv1)

for i in range(2, n + 1):
    csv_name2 = f"uniaxial_test_{i}.csv"
    csv2 = pd.read_csv(csv_name2, header=None, skiprows=5)
    data2.append(csv2)

# Load data F (N)
F = []

for j in range(0, n - 1):
    F_data = data1[j].iloc[:, 7].values * 1000
    F.append(F_data - F_data[0])

# Displacement data dL (mm)
dL = []

for k in range(0, n - 1):
    dL_data = data2[k].iloc[:, 1].values
    dL.append(dL_data - dL_data[0])
    
'''
UNIAXIAL TENSILE TEST ANALYSIS
------------------------------
'''

# Initial cross-sectional area (mm^2)
A_0 = 0.006 * 0.002

# Initial length (mm)
L_0 = 31.06 

# Initialize lists to store intersection points
σ_e_intersections = []
σ_t_intersections = []

# Loop through all experiments to compute intersection points
for i in range(n - 1):
    # Engineering Strain and Engineering Stress for the current experiment
    ε_e = dL[i] / L_0
    σ_e = F[i] / A_0
    
    # Ensure that σ_e and ε_e have the same length using interpolation
    min_len = min(len(σ_e), len(ε_e))
    
    # Create an evenly spaced range for interpolation
    x_new = np.linspace(0, 1, min_len)
    
    # Interpolate both arrays to match the same number of points (length)
    interpolate_ε_e = interp1d(np.linspace(0, 1, len(ε_e)), ε_e, kind='linear')
    interpolate_σ_e = interp1d(np.linspace(0, 1, len(σ_e)), σ_e, kind='linear')
    
    # Resample both arrays to have the same length
    ε_e_resampled = interpolate_ε_e(x_new)
    σ_e_resampled = interpolate_σ_e(x_new)

    # Use the first few resampled data points to calculate the gradient 
    σ_e_initial = σ_e_resampled[100:300] / 1e6
    ε_e_initial = ε_e_resampled[100:300]
    
    # Check if both arrays have the same length after interpolation
    if len(σ_e_initial) != len(ε_e_initial):
        raise ValueError(f"Length mismatch after interpolation: σ_e_initial has {len(σ_e_initial)} elements, ε_e_initial has {len(ε_e_initial)} elements.")
    
    # Calculate Engineering Young's modulus
    E_e = (σ_e_initial[-1] - σ_e_initial[0]) / (ε_e_initial[-1] - ε_e_initial[0])  
    print(f"E_e: {E_e / 1e3} GPa")

    # 0.2% offset line function for Engineering Stress-Strain
    def offset_line_e(ε_e):
        return E_e * (ε_e - 0.002)

    # Interpolation function for Engineering Stress-Strain curve
    stress_interpolator_e = interp1d(ε_e_resampled, σ_e_resampled, kind='linear', fill_value='extrapolate')

    # True Strain and True Stress for the current experiment
    ε_t = np.log(1 + ε_e_resampled)
    σ_t = σ_e_resampled * (1 + ε_e_resampled)

    # Use the first few data points to calculate the gradient 
    σ_t_initial = σ_t[100:300] / 1e6
    ε_t_initial = ε_t[100:300]
    
    # Check if both arrays have the same length after slicing
    if len(σ_t_initial) != len(ε_t_initial):
        raise ValueError(f"Length mismatch after slicing: σ_t_initial has {len(σ_t_initial)} elements, ε_t_initial has {len(ε_t_initial)} elements.")
    
    # Calculate True Young's modulus
    E_t = (σ_t_initial[-1] - σ_t_initial[0]) / (ε_t_initial[-1] - ε_t_initial[0]) 
    print(f"E_t: {E_t / 1e3} GPa")

    # 0.2% offset line function for True Stress-Strain
    def offset_line_t(ε_t):
        return E_t * (ε_t - 0.002)

    # Interpolation function for True Stress-Strain curve
    stress_interpolator_t = interp1d(ε_t, σ_t, kind='linear', fill_value='extrapolate')

    if i == 0:  # Change this to print only the graph for one specific experiment
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        # Engineering Stress-Strain plot
        ax1.plot(ε_e_resampled, σ_e_resampled / 1e6, label="Engineering Stress-Strain")
        ax1.plot(ε_e_resampled, offset_line_e(ε_e_resampled), 'g--', label="0.2% Offset Line (Eng)")  
        ax1.set_xlabel('Engineering Strain')
        ax1.set_ylabel('Engineering Stress (MPa)')
        ax1.set_ylim(0, 400)
        ax1.grid(True)
        ax1.legend()

        # True Stress-Strain plot
        ax2.plot(ε_t, σ_t / 1e6, label="True Stress-Strain")
        ax2.plot(ε_t, offset_line_t(ε_t), 'g--', label="0.2% Offset Line (True)")  
        ax2.set_xlabel('True Strain')
        ax2.set_ylabel('True Stress (MPa)')
        ax2.set_ylim(0, 400)
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()
