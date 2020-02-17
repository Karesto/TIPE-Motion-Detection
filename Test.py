import matplotlib.pyplot as plt
import cv2
import numpy as np
from Fonctions_auxiliaires import *

img = cv2.cvtColor(cv2.imread("j.png"), cv2.COLOR_BGR2GRAY)
a,b = img.shape

kernel1 = (255*3*3*np.ones((a-3,b-3)),255*3*3*np.ones((a-3,)),255*3*3*np.ones((b-3,)))
kernel2 = (np.zeros((a - 3, b - 3)), np.zeros((a - 3,)),np.zeros((b - 3,)))

res1 = quick_dilatation(img,3,kernel2)
res2 = quick_dilation(img,3,kernel1)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
res3 = cv2.dilate(img,kernel,iterations = 1)

myero = quick_erosion(img,3,kernel1)
thero = cv2.erode(img,kernel)


