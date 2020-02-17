import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

def Gaussi_vide (img):        # Function : Renvoie un tableau de la forme l*h*2 ( 2 pour moyenne et ec )
    l  = img.shape[0]
    c  = img.shape[1]
    A = np.array([0,0])
    Gaussi = [] 
    for i in range (l):
        Gaussi.append([])
        for j in range (c):
            Gaussi[i].append(np.copy(A))
    return(np.array(Gaussi))  
      
                
def moyen (fond,i,j):  #niveau de gris pour un seul pixel
    if len(fond.shape) == 2:
        return(fond[i][j])
    else :
        return(1/3*(fond[i][j][0])+ 1/3*fond[i][j][1]+1/3*fond[i][j][2])

def initi_gauss (Fond1,Gaussi):
    for i in range (len(Gaussi)):
        for j in range (len(Gaussi[0])):
            Gaussi[i][j] = [moyen(Fond1,i,j),0]
        
def maj_gaussi (Img, Gaussi,n):
    for i in range (len(Gaussi)):
        for j in range (len(Gaussi[0])):
            mu = Gaussi[i][j][0]
            sig = Gaussi[i][j][1]
            i1= moyen(Img,i,j)
            Gaussi[i][j][0] = (n*mu + i1)/(n+1)
            Gaussi[i][j][1] = (n*sig+(i1-mu)**2)/(n+1)
    
    
def calque_mouvement_initial (img):
    a  = img.shape[0]
    b  = img.shape[1]
    VF = np.zeros((a, b))
    return(VF)

def MaJ_CM (VF,Gaussi,img):
    for i in range (len(Gaussi)):
        for j in range (len(Gaussi[0])):
            if (abs(moyen(img,i,j) -Gaussi[i][j][0]) > Gaussi[i][j][1]) : VF[i][j] = 255
            else : VF[i][j] = 0


def Code (): 
    video = cv2.VideoCapture('C:/Users/hass.chouk/Desktop/Codes/Vid_Test.avi')
    (est_ouvert,Fond) = video.read()
    if est_ouvert:
        #Fond = cv2.resize(Fond, (48, 48), 0, 0, cv2.INTER_CUBIC)
        gaussienne = Gaussi_vide(Fond)
        initi_gauss(Fond, gaussienne)
        n=0
        l=[]         #truc pour tester 
        CM = calque_mouvement_initial(Fond)
        
        while est_ouvert and n < 100:
            for i in range (3) : #test
                video.grab()
               
            (ouvert,img1) = video.read()
            #img1 = cv2.resize(img1, (48, 48), 0, 0, cv2.INTER_CUBIC)
            n+= 1
            name = 'img'+str(n)+'.jpg'
            print(n)     #test
            cv2.imwrite(name,CM)
            if ouvert:
                MaJ_CM(CM,gaussienne,img1)
                maj_gaussi(img1,gaussienne, n)


                l.append(np.copy(CM))
        return(l)
                
            
    

        

