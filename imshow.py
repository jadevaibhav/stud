from matplotlib import pyplot as plt
import numpy as np

img = np.load("check.npy")
plt.imshow(img[0].transpose((1,2,0))[...,::-1])
plt.show()