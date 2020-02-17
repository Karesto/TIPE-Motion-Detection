import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def Moyenne_Sigma (image):
    '''Initialisation du code'''
    return(np.copy(image),np.zeros(image.shape))



def rendu(moyenne,sigma,image):
    return(np.greater(np.abs(moyenne-image),2*sigma))



def Code(code_name):
    '''
    -Code_name: Nom de la méthode utilisé
    -List : Liste des BDD utilisée en format (nom, nbr d'images)
    -path_in : Chemin d'accès 
    '''
    list = [("boulevard",2500), ("bungalows",1700), ("canoe",1189) ,("highway",1700), ("parking",2500)] #Voir premier com'
    
    for elm in list :
        '''Setup des chemins'''
        file,nbr = elm
        path_in =r"E:\Users\Ahmed\Desktop\TIPE\BDD\\" + file + "\input"  #Chemin d'accès
        path_out = r"E:\Users\Ahmed\Desktop\TIPE\Results\\"  + code_name + "\\" +file   #Chemin d'enregistrement
        os.makedirs(path_out, exist_ok=True)
        
        '''Le Code''' 
        
        Fond =cv2.cvtColor(cv2.imread(path_in + "\img1.jpg"),cv2.COLOR_BGR2GRAY) #Première image comme fond
        moyenne,sigma = Moyenne_Sigma(Fond)
        for i in range (2,nbr+1):
            print(i)
            image = cv2.cvtColor(cv2.imread(path_in + "\img"+ str(i) +".jpg"),cv2.COLOR_BGR2GRAY)
            name = '_rendu'+str(i)+'.jpg'
            moyenne = (i*moyenne + image)/(i+1)     #Mise à jour de la moyenne
            sigma = np.sqrt((i*(sigma**2) + (image-moyenne)**2)/(i+1)) #Mise a jour de la gaussienne
            cv2.imwrite(path_out  + "\img"  + name,255*(rendu(moyenne,sigma,image)))  #Ecriture de l'image rendu sur le disque
        print(moyenne)
        cv2.imwrite(path_out + "\Fond.jpg",moyenne)
    return()