import numpy as np
import cv2
########################### Codes Auxiliaires ##########################
'''A faire: Revoir les érosies (fonctionnement avec du blanc et s'assurer que tout va bien





'''
def conv2 (img, prm,n=3):
    '''
    Deuxième type de convolution, Matrices n*n autour du point, tout les pixels autour ont le même poids,les bords restent inchangés
    img: Matrice de l'image qu'on veut "convolution"
    prm: Réel entre 0 et 1 
    n : entier impair, taille de la matrice de convolution , 
    '''
    a,b = img.shape
    res = np.copy(img)

    prm1 = (1-prm)/8
    res[1:a-1,1:b-1]=prm*img[1:a-1,1:b-1]+ (prm1*img[0:a-2,1:b-1]+ prm1*img[2:a,1:b-1]+ prm1*img[1:a-1,0:b-2]+ prm1*img[1:a-1,2:b]) + (prm1*img[2:a,0:b-2] + prm1*img[0:a-2,0:b-2] +prm1*img[0:a-2,2:b]+ prm1*img[2:a,2:b])
    return(res)
    
    
def erosion(img,kernel,n):
    ''' n impair uniquement , taille du kernel de travail'''
    ''' kernel : matrice d'érosion -->   kernel = 255*np.ones((n,n),np.unit8) il est inclu car ça diminue la complexité de le réinclure '''
    a,b = np.shape(img)
    d= n//2

    res = np.zeros((a,b), np.unit8)
    for i in range (d,a-d):
        for j in range (d,b-d):
            if img[i-d:i+d,j-d,:j+d] == kernel : res[i,j] = 255
    return(res)


def sommation(img):
    '''
    sommation(img) renvoie une image res telle que res[i][j] est la somme de la sous matrice img[i:a][j:b]
    img: image a sommer
    '''
    a,b = np.shape(img)
    res = img.astype(numpy.int32)
    #Le astype sert pour ne pas avoir du module 255

    for i in range (a-2,0):
        res[i,b-1] += res[i+1,b-1]
    for j in range (b-2,0):
        res[i,a-1] += res[i+1,a-1]
    for i in range (a-2,0):
        for j in range (b-2,0):
            res[i,j] += res[i+1,j] + res[i,j+1] - res[i+1,j+1]
    return(res)
    


def conv2_gen(img,prm,n):

    sum = sommation(img)
    a,b = np.shape(img)
    res = img.astype(np.float)
    d= n//2
    for i in range (d,a-d-1):
        for j in range (d,b-d-1):
            res[i,j] = (sum[i-d,j-d] - sum[i+d+1,j-d] - sum[i-d,j+d+1] + sum[i+d+1,j+d+1])*(1-prm)/(n*n) -(1-2*prm)/(n*n-1)*img[i,j]
    for i in range (d,a-d-1):
        res[i,b-d-1] = (sum[i-d,b-n] - sum[i+d+1,b-n])*(1-prm)/(n*n) -(1-2*prm)/(n*n-1)*img[i,b-d-1]
    for j in range (d,b-d-1):
        res[a-d-1,j] = (sum[a-n,j-d] - sum[a-n,j+d+1])*(1-prm)/(n*n) -(1-2*prm)/(n*n-1)*img[a-d-1,j]
    res[a-1-d,b-1-d] = (sum[a-n,b-n])*(1-prm)/(n*n) -(1-2*prm)/(n*n-1)*img[a-1-d,b-1-d]
    return(res)
    
#A vectorialiser avec numpy, se fait bien je dirais (surtotu si on utilise la fonction d'en bas)
    

def quick_erosion(img,n):
    '''Hmm, are theses 'indices' okay ?'''
    sum = sommation(img)
    a,b = np.shape(img)
    res = img.astype(np.float)
    d=n//2
    kernel1 = n*n*np.ones((a-n,b-n))
    kernel2 = n*n*np.ones((b-n,1))
    kernel3 = n*n*np.ones((1,a-n))
    
    res[d:a-d-1,d:b-d-1] = (np.equal((sum[:a-n,:b-n] - sum[:a-n,n:b-1] - sum[n:a-1,:b-n] + sum[n:a-1,n:b-1]),kernel1)).astype(np.float)
    #Avant dernière colonne
    res[d:a-d-1,b-d-1] = (np.equal(sum[:a-n-1,b-n]-sum[n:a,b-n],kernel2)).astype(np.float)
    #Avant dernière ligne
    res[a-d-1,d:b-d-1] = (np.equal(sum[a-n,:b-n-1]-sum[a-n,n:b],kernel3)).astype(np.float)
    #Point du coin
    res[a-1-d,b-1-d] = ((sum[a-n,b-n]) == n*n).astype(np.float)
    return(res)


def quick_convoution(img,prm,n):
    sum = sommation(img)
    a,b = np.shape(img)
    res = img.astype(np.float)
    d=n//2
    res[d:a-d-1,d:b-d-1] = ((sum[:a-n,:b-n] - sum[:a-n,n:b-1] - sum[n:a-1,:b-n] + sum[n:a-1,n:b-1])*(1-prm)/(n*n) -(1-2*prm)/(n*n-1)*img[i,j]).astype(np.float)
    #Avant dernière colonne
    res[d:a-d-1,b-d-1] = ((sum[:a-n-1,b-n]-sum[n:a,b-n])*(1-prm)/(n*n) -(1-2*prm)/(n*n-1)*img[i,j]).astype(np.float)
    #Avant dernière ligne
    res[a-d-1,d:b-d-1] = (np.equal(sum[a-n,:b-n-1]-sum[a-n,n:b]*(1-prm)/(n*n) -(1-2*prm)/(n*n-1)*img[i,j])).astype(np.float)
    #Point du coin
    res[a-1-d,b-1-d] = (sum[a-n,b-n])*(1-prm)/(n*n) -(1-2*prm)/(n*n-1)*img[a-1-d,b-1-d]

#On utilisera la fonction dilation de OpenCV pour cette opération (car j'ai la flemme de coder un truc moche)
#dilation = cv2.dilate(img,kernel,iterations = 1)
#On pourra utiliser opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel), c'est une erosion puis dilation
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            