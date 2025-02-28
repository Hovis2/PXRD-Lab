
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:36:27 2025

@author: victo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import scipy

h = scipy.constants.h # Js
c = scipy.constants.c # m/s
E = (17.45 * 10**3) * 1.6021766339999e-19 # J
wavelength = (h * c) / E  # m

# Read data file
df = pd.read_csv("/Users/victor/Desktop/Sample1.txt", delim_whitespace=True, header=None, index_col=False)
df.columns = ["Column1", "Column2"]

df2 = pd.read_csv("/Users/victor/Desktop/Sample2.txt", delim_whitespace=True, header=None, index_col=False)
df2.columns = ["Column1", "Column2"]

# Convert to radians from degrees
rad = np.radians(df["Column1"]/2)
rad2 = np.radians(df2["Column1"]/2)

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
        

def d(angle):
    return wavelength / (np.sin(angle) * 2)

#FCC Spectrum
theta_values = [0.15393, 0.17765, 0.25161, 0.29557, 0.30882]
multipliers = [np.sqrt(3), 2, np.sqrt(8), np.sqrt(11), np.sqrt(12)]

# Compute d-values and apply multipliers
d_values = [d(theta) * m for theta, m in zip(theta_values, multipliers)]

# Sum and average
d_sum = sum(d_values)
d_avg = d_sum / len(d_values)

print(f"Average of Fcc: {d_avg}")
    
#Bcc
theta_values2 = [0.17765, 0.25231, 0.30952]
multipliers2 = [np.sqrt(2), 2, np.sqrt(6)]

# Compute d-values and apply multipliers
d_values2 = [d(theta) * m for theta, m in zip(theta_values2, multipliers2)]

# Sum and average
d_sum2 = sum(d_values2)
d_avg2 = d_sum2 / len(d_values2)

print(f"Average of Bcc: {d_avg2}")

#crystallite size
def t(Beta, angle):
    return (0.94 * wavelength) / (Beta * np.cos(angle))

#Fcc
FWHM = np.radians([4.58037, 4.98198, 4.80901, 5.20054, 4.71695])
theta_values = [0.15393, 0.17765, 0.25161, 0.29557, 0.30882]
t_values = [t(Beta, angle) for Beta, angle in zip(FWHM, theta_values)]
    
# Sum and average
t_sum = sum(t_values)
t_avg = t_sum / len(t_values)


print(f"Average t of Fcc: {t_avg}")

#Bcc
FWHM2 = np.radians([3.91045, 4.02491, 4.99830])
theta_values2 = [0.17765, 0.25231, 0.30952]

t_values2 = [t(Beta, angle) for Beta, angle in zip(FWHM2, theta_values2)]
    
# Sum and average
t_sum2 = sum(t_values2)
t_avg2 = t_sum2 / len(t_values2)


print(f"Average t of Bcc: {t_avg2}")

