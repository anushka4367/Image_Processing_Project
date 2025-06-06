# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 00:03:51 2024

@author: anushka
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

# Load the images
image_files = {
    '296K': 'images/img_296k.png',
    '320K': 'images/img_320k.png',
    '329K': 'images/img_329k.png',
    '337K': 'images/img_337k.png'
}
images = {temp: cv2.imread(file) for temp, file in image_files.items()}

# Convert images to grayscale
gray_images = {temp: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for temp, img in images.items()}

# Threshold the initial image to get binary image
initial_temp = '296K'
initial_img = gray_images[initial_temp]
_, initial_binary = cv2.threshold(initial_img, 100, 255, cv2.THRESH_BINARY_INV)

# Find connected components in the initial binary image
initial_labels = label(initial_binary)
initial_props = regionprops(initial_labels)

# Number of particles (same for all temperatures)
num_particles = len(initial_props)

# Dictionary to store the count of dark and bright particles
particle_counts = {
    '296K': {'dark': num_particles, 'bright': 0},
    '320K': {'dark': 0, 'bright': 0},
    '329K': {'dark': 0, 'bright': 0},
    '337K': {'dark': 0, 'bright': 0}
}

# Process each image
for temp, img in gray_images.items():
    if temp == initial_temp:
        continue

    # Threshold the current image to get binary image
    _, current_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Compare current binary image with initial binary image
    bright_particles_count = 0
    
    for prop in initial_props:
        # Get coordinates of the current particle
        coords = prop.coords
        # Check the intensity in the current image
        if np.mean(current_binary[coords[:, 0], coords[:, 1]]) < 128:
            bright_particles_count += 1
    
    dark_particles_count = num_particles - bright_particles_count
    particle_counts[temp]['dark'] = dark_particles_count
    particle_counts[temp]['bright'] = bright_particles_count

# Extract data for plotting
temperatures = ['296K', '320K', '329K', '337K']
dark_counts = [particle_counts[temp]['dark'] for temp in temperatures]
bright_counts = [particle_counts[temp]['bright'] for temp in temperatures]

# Create the histogram
fig, ax = plt.subplots()

bar_width = 0.4
x = np.arange(len(temperatures))

ax.bar(x, dark_counts, bar_width, label='Dark Particles', color='blue')
ax.bar(x, bright_counts, bar_width, bottom=dark_counts, label='Bright Particles', color='orange')

ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Number of Particles')
ax.set_title('Particle Count vs. Temperature')
ax.set_xticks(x)
ax.set_xticklabels(temperatures)
ax.legend()

plt.show()
