############################################---Code Principal---########################################


## A faire : Fix convolution temporelle pour qu'elle donne autant de résultat que les autres







import numpy as np
import cv2
import os 


def conv1 (img, prm,n=3):
    '''
    Premier type de convolution, Matrices n*n autour du point, plus on s'éloigne, plus les poids sont faible, les bords restent inchangés
    img: Matrice de l'image qu'on veut "convolution"
    prm: Réel entre 0 et 1 
    n : entier impair, taille de la matrice de convolution , Trop compliquée pour ce type de convolution qu'on laissera a un 3x3 :)
    '''
    a,b = img.shape
    d=n//2
    res = np.copy(img)
    prm1 = (1-prm)/6
    prm2 = (1-prm)/12
    res[1:a-d,d:b-1]=prm*img[1:a-1,1:b-1]+ (prm1*img[0:a-2,1:b-1]+ prm1*img[2:a,1:b-1]+ prm1*img[1:a-1,0:b-2]+ prm1*img[1:a-1,2:b]) + (prm2*img[2:a,0:b-2] + prm2*img[0:a-2,0:b-2] +prm2*img[0:a-2,2:b]+ prm2*img[2:a,2:b])
    return(res)
    
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

def Moyenne_Sigma (image):
    '''
    Initialisation de la matrice de gaussiennes
    Image: Matrice de l'image, ne sert qu'a prendre les dimensions.
    Initialisation très pauvre
    '''
    return(np.copy(image),np.zeros(image.shape))



def rendu(moyenne,sigma,image,n=1):
    '''
    Renvoie le rendu pour le modèle probabilsite des gaussienens passées en paramètre 
    
    moyenne: Matrice de Réels, Moyenne des gaussiennes pour chaque pixel
    sigma: Matrice de Réel, écart-type des gaussiennes pour chaque pixel
    image : Image a traiter
    '''
    return((np.abs(moyenne-image)>=n*sigma).astype(np.uint8))
    
'''Code Noémie, écrit sur codes auxiliaires.py mais non débuggé'''



def Carres(source,taille_masque=3,seuil=50):
    
#source=str de l'image, ,taille_masque=entier impair, seuil entre 0 et 155 a modifier si c'est pas beau

    def arrondit(img,seuil):  ##
        ret, mask = cv2.threshold(img, seuil, 255, cv2.THRESH_BINARY)
        return(mask)
        
    def remplit(test,tm):  ##
        masque= np.zeros((tm,tm))
        (a,b)=(int(tm/2),1+int(tm/2))
        (ta,tb)=np.shape(test)
        res_tem=np.zeros((ta,tb))
        res_tem=255+res_tem
        for i in range(a,ta-b):
            for j in range(a,tb-b):
                if test[i,j]== 0:
                    res_tem[i-a:i+b,j-a:j+b]=masque
        for i in range(a,ta-b):
            for j in range(a,tb-b):
                if np.array_equal(res_tem[i-a:i+b,j-a:j+b],masque):
                    test[i,j]=0
                else:
                    test[i,j]=255
        return(test)
        
    img=np.copy(source)
    test=arrondit(img,seuil)
    return(remplit(test,taille_masque))



#La fonction principale du code, Elle s'occupe de la comparaison et de l'envoie d'un résultat après comparaison des fonds.
#Caractéristiques : Les images sont traitées en noir en blanc uniquement
#Utilise les fonctions : Conv1 ou Conv1, Carres, Rendu

def Code(code_name, prm_conv = 1, conv=conv2, avant = False, apres = False, filtre = False, conv_tempo = 1, prm_gauss=1 ):
    '''
    -Code_name: Nom de la méthode utilisé, un dossier sera créé (le chemin est précisé pour mon PC mais peut être changé plus bas
    -prm_conv : Réel entre 0 et 1 Paramète de la convolution (pourra être dédoublé pour la convolution avant/convolution après)
    -Conv : Fonction , C'est la fonction de convolution a utiliser, on utilisera uniquement une seule pour les deux convolutions avant et après, fonctions disponible : conv1, conv2
    -avant : Booléen pour déterminer si on utilise la convolution sur les images avant de les comparer
    -après : Booléen pour déterminer si on utilise la convolution sur les images après les avoir comparé
    -filtre : Booléen pour déterminer si on utilise la méthode du remplissage de carrés (après comparaison)
    '''
    
    
    #Liste des dossiers (nom,nombre d'images) qui permet d'ajouter/supprimer rapidement des bases de données qu'on utilise
    base = [("boulevard",2500), ("bungalows",1700), ("canoe",1189) ,("highway",1700) ,("parking",2500)]
    
    #Première boucle pour utiliser toutes les bases passées dans la liste "base"
    for elm in base :
        #Setup des chemins 
        #On crée  les chemins si ils manquent, avec les noms adaptés
        file,nbr = elm
        path_in =r"E:\Users\Ahmed\Desktop\TIPE\BDD\\" + file + "\input"  #Chemin d'accès
        path_out = r"E:\Users\Ahmed\Desktop\TIPE\Results\\"  + code_name + "\\" +file   #Chemin d'enregistrement
        os.makedirs(path_out, exist_ok=True)
        #Fichier Paramètre pour sauvegarder les paramètres utilisés
        parametres = "code_name :" + str(code_name) +"\n" +\
                     "prm_conv:"+str(prm_conv)+ "\n" +\
                     "conv:" + str(conv) +"\n" +\
                     "convolution spaciale avant:" + str(avant) +  "\n" +\
                     "convolution spaciale après:" + str(apres) + "\n" +\
                     "filtre:" + str(filtre) + "\n" +\
                     "conv_tempo:" + str(conv_tempo) + "\n" +\
                     "prm_gauss" + str(prm_gauss) + "\n"  
        '''
        -----------------Le Code-------------------
        C'est ici ou on rassemble de tout les morceaux
        
        ''' 
        Fond =cv2.cvtColor(cv2.imread(path_in + "\img1.jpg"),cv2.COLOR_BGR2GRAY) #Première image comme fond
        moyenne,sigma = Moyenne_Sigma(Fond)      #On initialise la matrice de gaussiennes avec la fonction Moyenne_Sigma()
        
        #Convolution Temporelle: Tentative"Co
        n = conv_tempo
        for i in range (2,(nbr+1)//n):
            image= image = (1/n)*cv2.cvtColor(cv2.imread(path_in + "\img"+ str(i) +".jpg"),cv2.COLOR_BGR2GRAY)
            #initialisation de la convolution (s'il ya)
            for j in range (1,n) :
                image= (1/n)*cv2.cvtColor(cv2.imread(path_in + "\img"+ str(i+j) +".jpg"),cv2.COLOR_BGR2GRAY)
            #print(image.dtype)
            #print(i,image)  Utilisée pour débugger le code
            #Utilisation de la convolution avant le traitement
            if avant : image = conv(image, prm_conv)     
            #Nom du fichier a enregistrer
            name = '_rendu'+str(i)+'.jpg'
            #Mise a jour de la moyenne de la gaussienne
            moyenne = (i*moyenne + image)/(i+1)
            sigma = np.sqrt((i*(sigma**2) + (image-moyenne)**2)/(i+1))
            #Traitement/comparaison : utilisation de la fonction rendu()
            res = (rendu(moyenne,sigma,image,prm_gauss))
            #Utilisation la convolution après le traitement
            if apres : res = conv(res,prm_conv)      
            #Utilisation du remplissage des carrés
            if filtre : res = filtre(res)
            #Ecriture des résultats sur les disques
            cv2.imwrite(path_out  + "\img"  + name , 255*res)
            
        
        cv2.imwrite(path_out + "\Fond.jpg",255*moyenne)
    f = open(r"E:\Users\Ahmed\Desktop\TIPE\Results\\"  + code_name + "\\" + "Paramètres.txt",'w')
    f.write(parametres)
    f.close()
    return()
    
    
    
    
    
    
    
    
    