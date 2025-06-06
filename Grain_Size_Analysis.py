# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:33:51 2024

@author : anushka
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color

img = cv2.imread("images/grains2.jpg", 0)
cv2.imshow("Grains Image", img)
cv2.waitKey(0)

# Convert pixel to nanometer (nm)
pixels_to_nm = 0.5  # 1 pixel = 500 nm

ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(thresh, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)

mask = dilated == 255

s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
labeled_mask, num_labels = ndimage.label(mask, structure=s)
img_label_overlay = color.label2rgb(labeled_mask, image=img, bg_label=0)
cv2.imshow("Labeled Grains", img_label_overlay)
cv2.waitKey(0)
clusters = measure.regionprops(labeled_mask, img)

print("\tArea(pixels)\tArea (nm^2)")

for prop in clusters:
    grain_area_pixels = prop.area
    grain_area_nm2 = grain_area_pixels * (pixels_to_nm ** 2)
    print('{}\t\t{}\t\t{}'.format(prop.label, grain_area_pixels, grain_area_nm2))

# Calculate grain size in nanometers (nm)
grain_sizes_nm = []

for prop in clusters:
    grain_radius_pixels = np.sqrt(prop.area/np.pi) # Calculate radius in pixels
    grain_radius_nm = grain_radius_pixels * pixels_to_nm  # Convert radius to nm
    grain_size_nm = (4/3) * np.pi * (grain_radius_nm**3)  # Calculate grain size in nm^3

    grain_sizes_nm.append(grain_size_nm)



# Plot histogram with grain sizes in nm
plt.figure()
plt.hist(grain_sizes_nm, bins=10000)
plt.xlabel('Grain Size(nm^3)')
plt.ylabel('Count')
plt.title('Grain Size Distribution')
plt.grid(True)
plt.xlim(0, 250)
plt.show()

# Close all OpenCV windows
cv2.destroyAllWindows()
