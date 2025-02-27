# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:36:27 2025

@author: victo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

h = 6.626 * 10**(-34) # Js
c = 299792458 # m/s
E = (17.45 * 10**3) / (6.242 * 10**18) # J
wavelength = (h * c) / E  # m

# Read data file
df = pd.read_csv("C:/Users/victo/Desktop/Sample1.txt", delim_whitespace=True, header=None, index_col=False)
df.columns = ["Column1", "Column2"]

df2 = pd.read_csv("C:/Users/victo/Desktop/Sample2.txt", delim_whitespace=True, header=None, index_col=False)
df2.columns = ["Column1", "Column2"]

# Convert to radians from degrees
rad = df["Column1"]/2 * np.pi / 180
rad2 = df2["Column1"]/2 * np.pi / 180

# Define the xmin and max
x_min1, x_max1 = 0.1, rad.max() 

# Finding the peaks and FWHM then plotting
peaks, _ = find_peaks(df["Column2"], height=4)  # Height is the threshold for the peak to be marked
widths, width_heights, left_ips, right_ips = peak_widths(df["Column2"], peaks, rel_height=0.5) #FWHM

# Plot data with FWHM
plt.plot(rad, df["Column2"], label="Spectrum 1")
plt.plot(rad.iloc[peaks], df["Column2"].iloc[peaks], "rx", label="Peaks")  # Mark peaks
plt.hlines(width_heights, rad.iloc[left_ips.astype(int)], rad.iloc[right_ips.astype(int)], color="green", label="FWHM") #Mark FWHM
plt.ylim(0, 50) 
plt.xlim(0.1,)
plt.xlabel("Theta")
plt.ylabel("Intensity")
plt.legend()
plt.show()

# Repeat for second dataset
x_min, x_max = 0.15, rad2.max() 

peaks2, _ = find_peaks(df2["Column2"], height=3.4)
widths2, width_heights2, left_ips2, right_ips2 = peak_widths(df2["Column2"], peaks2, rel_height=0.5)

plt.plot(rad2, df2["Column2"], label="Spectrum 2")
plt.plot(rad2.iloc[peaks2], df2["Column2"].iloc[peaks2], "rx", label="Peaks")
plt.hlines(width_heights2, rad2.iloc[left_ips2.astype(int)], rad2.iloc[right_ips2.astype(int)], color="green", label="FWHM")
plt.ylim(0, 13)
plt.xlim(0.15,)
plt.xlabel("Theta")
plt.ylabel("Intensity")
plt.legend()
plt.show()

# Print FWHM values
for i, peak in enumerate(peaks):
    peak_position1 = rad.iloc[peak]
    # Check if the peak is within the xlim range
    if x_min1 <= peak_position1 <= x_max1:
        print(f"(Spectrum 1): Position = {peak_position1:.5f}, Height = {df['Column2'].iloc[peak]:.5f}, FWHM = {widths[i]:.5f}")

#Repeat
for i, peak in enumerate(peaks2):
    peak_position = rad2.iloc[peak]
    if x_min <= peak_position <= x_max:
        print(f"(Spectrum 2): Position = {peak_position:.5f}, Height = {df2['Column2'].iloc[peak]:.5f}, FWHM = {widths2[i]:.5f}")