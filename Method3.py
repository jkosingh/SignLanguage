import numpy as np
import cv2
from matplotlib import pyplot as plt

from skimage import io, color

rgb = io.imread('D:\\dataPath.jpg')
img = color.rgb2lab(rgb)
thresholded = np.logical_and(*[img[..., i] > t for i, t in enumerate([40, 0, 0])])
from matplotlib import pyplot as plt
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(rgb);         ax[0].axis('off')
ax[1].imshow(thresholded); ax[1].axis('off')
plt.show()