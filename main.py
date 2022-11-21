# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# https://scikit-image.org/docs/stable/user_guide/getting_started.html

import os

import matplotlib.pyplot as plt
import skimage
from skimage import io

filename = os.path.join(skimage.data_dir, 'fotos/bios/GUU-Draco-Amo9-29Mai70-515-29,1-2-0,5-DIR.jpg')
moon = io.imread('C://repos/msc_ds_vc/photos/bios/GUU-Draco-Amo9-29Mai70-515-29,1-2-0,5-DIR.jpg')
print(moon.shape)
plt.imshow(moon)

