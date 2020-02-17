import numpy as np
import cv2
from Fonctions_auxiliaires import *
import matplotlib.pyplot as plt


img = cv2.cvtColor(cv2.imread(r"E:\Users\Ahmed\Desktop\TIPE\Project\Codes\j.png"),cv2.COLOR_BGR2GRAY)
a,b = img.shape
res3 = quick_convolution(img,1/9,3)
res1 = conv2(img,1/9,3)
res2 = conv2_generalisee(img,1/9,3)
kernel = 1/9*np.ones((3,3))
res4 = cv2.blur(img,(3,3))

show =r
cv2.imshow("window",show)
plt.imshow(show)
plt.show()