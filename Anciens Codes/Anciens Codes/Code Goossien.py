import numpy as np
import matplotlib.pyplot as plt
import cv2

def Gaussi (img):
    a = img.shape(0)
    b = img.shape(1)
    A = np.array([0,0])
    Gaussi = [] 
    for i in range (a):
        Gaussi.append([])
        for j in range (b):
            Gaussi[i].append(np.copy(A))
            

def moyen (fond,i,j):
    if len(fond.shape) == 2:
        return(fond[i][j])
    else :
        return(1/3*(fond[i][j][0]+fond[i][j][1]+fond[i][j][2]))

def initi_gauss (Fond1,Gaussi):
    for i in range (len(Gaussi)):
        for j in range (len(Gaussi[0])):
            Gaussi[i][j] = [moyen(Fond1,i,j),25]
        
def maj_gaussi (Img, Gaussi,n):
    for i in range (len(Gaussi)):
        for j in range (len(Gaussi[0])):
            V = Gaussi[i][j][0]
            U = Gaussi[i][j][1]
            i1= moyen(Img,i,j)
            Gaussi[i][j][0] = (n*V+(i1-U)**2)/(n+1)
            Gaussi[i][j][1] = (n*U + i1)/(n+1)
    

def Background_Blend (Background, New_Img, alpha):
    A = Background * alpha + New_Img*(1-alpha)
    A=A.astype(np.uint8)
    #A=cv2.addWeighted(Background,alpha,New_Img,1-alpha,0)
    return(A)
    

def Code (): 
    video = cv2.VideoCapture('Vid_Test.avi')
    (booli,Fond) = video.read()

    n=0
    
    while booli:
        video.grab()
        video.grab()
        (booli,img1) = video.read()
        n+= 1
        name = 'img'+str(n)+'.jpg'
        
        if booli:
            maj_gaussi(img1,Gaussi, n)

        
        

