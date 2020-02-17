import random
import numpy as np
from Fonctions_auxiliaires import *
import cv2


matrice = np.array([[random.choice([0, 0, 0, 1, 1, 1, 1, 1]) for i in range(11)] for k in range(11)])
matrice = 255*np.array([[0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0],
       [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
       [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
       [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
       [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
       [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
       [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
       [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
       [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1]])

# matrice = np.random.rand(5,5)
# matrice = (255*matrice).astype(np.uint)
# matrice2 = np.array([[229 ,82  ,86 ,132 ,174] ,[172 , 54, 120, 234 ,138],[ 44 ,176,  44, 227 ,194], [193, 60 ,124 , 41 , 68], [  8 , 28,  94 ,184 , 99]])
# matrice3=np.array([[179 , 36 ,235, 190, 122],[211, 198 , 12  ,95 , 94],[195  ,74 ,  1 , 71, 221],[123 , 10 ,251 ,107 ,112],[ 91 ,184 ,157, 103 , 68]])

#
# matrice = (np.abs(matrice2-matrice3) >50)*1
# print(matrice)

n = len(matrice)
a =11
b= 11
ero = 3
kernel1 = 255 * 3 * 3 * np.ones((a - ero, b - ero))
kernel2 = 255 * 3 * 3 * np.ones((a - ero,))
kernel3 = 255 * 3 * 3 * np.ones((b - ero,))

matrice = quick_erosion(matrice, ero , (kernel1,kernel2,kernel3), coeff=0.6 )


print(matrice)



for i in range(len(matrice)):
    for j in range(len(matrice)):

        if matrice[i,j] == 0:
            print("""\\fill[black!80] (""" + str(j) + """,""" + str(n-i-1) + """) rectangle (""" + str(j + 1) + """,""" + str(n -i) + """) ;""")




#
#
# '''
# string = "\\node[inner sep=0.1cm,outer sep=0pt,anchor=center, minimum size=1cm] at "
#
#
# n = len(matrice)
#
# for i in range(len(matrice)):
#
#     for j in range(len(matrice)):
#
#         print(string + str((j + 0.5,n-i-0.5)) + " {" + str(matrice[i,j]) + "};")
#
# '''
