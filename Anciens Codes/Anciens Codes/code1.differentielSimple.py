import numpy as np
import matplotlib.pyplot as plt
import cv2


video = cv2.VideoCapture('Vid_Test.avi')
video.grab()
a=video.isOpened()
k=0

while a:
    (a,img1)=video.retrieve()
    video.grab()
    video.grab()
    (a,img2) = video.retrieve()
    k+= 1
    name = 'img'+str(k)+'.jpg'
    if a:
        res1= cv2.absdiff(img1, img2)
        cv2.imwrite(name,res1)
#Code deux avec fond
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
    
            #res1=np.abs(Fond- img1)
            print(Fond, img1.shape)
            
            
            res1= cv2.absdiff(Fond, img1)
            cv2.imwrite(name,res1)
            Fond = Background_Blend(Fond, img1,0.9)  




























'''
a= cv2.imread('1.png', 0)
b= cv2.imread('2.png', 0)
c= cv2.imread('3.png', 0)


res1= cv2.absdiff(a, b)
res2= cv2.absdiff(b,c)
resbis= cv2.bitwise_and(res1, res2)

plt.imshow(resbis)
plt.show()'''