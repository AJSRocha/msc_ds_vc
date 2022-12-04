import os
import matplotlib.pyplot as plt
import skimage
import numpy as np
from scipy import ndimage as ndi
from skimage import io
import pandas as pd
from pygrowthmodels import vonbertalanffy
import scipy
import cv2




teste_05 = io.imread('photos/bios/GUU-Deneb-Amo2-21Jan08-49-20,9-2-0,5.jpg')
teste_06 = io.imread('photos/bios/GUU-Deneb-Amo2-21Jan08-54-21-2-0,5.jpg')

hpf = teste_06 - cv2.GaussianBlur(teste_06,(101,101),0) + 0

plt.imshow(hpf, cmap=plt.cm.gray)
plt.show()

plt.imshow(teste_06)
plt.show()

plt.imshow((1 * hpf) + teste_06, cmap=plt.cm.gray)
plt.show()

