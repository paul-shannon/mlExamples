import matplotlib.pyplot as plt
import numpy as np
bitmap = np.fabs(np.random.randn(100, 100))
np.min(bitmap)
np.max(bitmap)
bitmap = bitmap - np.min(bitmap)
bitmap = bitmap/np.max(bitmap)
bitmap[:, 45:55] = 1
img = plt.imshow(bitmap)
plt.show() 
