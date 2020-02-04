# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------------------------------
def showTwoMatricesAsBitMaps(m1, m2):

   fig, (p1, p2) = plt.subplots(1,2)
   img1 = p1.imshow(m1)
   img2 = p2.imshow(m2)
   fig.show()
    
#------------------------------------------------------------------------------------------------------------------------
m1 = np.fabs(np.random.randn(100, 100))
m1 = m1 - np.min(m1)
m1 = m1/np.max(m1)
m2 = np.matrix.copy(m1)
m1[:, 45:55] = 1
m2[45:55, :] = 1
showTwoMatricesAsBitMaps(m1, m2)
