import numpy as np
import matplotlib.pyplot as plt
import cv2
import os



def Background_Blend (Background, New_Img, alpha):
    A = Background * alpha + New_Img*(1-alpha)
    A=A.astype(np.uint8)
    #A=cv2.addWeighted(Background,alpha,New_Img,1-alpha,0)
    return(A)
    

video = cv2.VideoCapture('Vid_Test.avi')
(booli,Fond) = video.read()

k=0

while booli:
    video.grab()
    video.grab()
    (booli,img1) = video.read()
    k+= 1
    name = 'img'+str(k)+'.jpg'
    
    if booli:

        #res1=np.abs(Fond- img1)
        print(Fond, img1.shape)
        
        
        res1= cv2.absdiff(Fond, img1)
        cv2.imwrite(name,res1)
        Fond = Background_Blend(Fond, img1,0.9)
    
        

