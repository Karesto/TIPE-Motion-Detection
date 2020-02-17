########################################---Code Principal---########################################
'''
-prm_conv    : Réel entre 0 et 1 Paramète de la convolution (pourra être dédoublé pour la convolution avant/convolution après)
-Conv        : Fonction , C'est la fonction de convolution a utiliser, on utilisera uniquement une seule pour les deux convolutions avant et après, fonctions disponible : conv1, conv2
-avant       : Booléen pour déterminer si on utilise la convolution sur les images avant de les comparer
-après       : Booléen pour déterminer si on utilise la convolution sur les images après les avoir comparé
-filtre      : Booléen pour déterminer si on utilise un filtre morphologique (auquel cas il est inséré a la place du booléen)
-n_filtre    : Taille du Noyau du filtre
-kernel      : Matrice pour le filtre
i_iteration  : Nb d'application du filtre morphologique
-conv_tempo  : Nombre d'images a moyenner pour la convolution temporelle
-prm_gauss   : Paramètre qui se multiplie a l'écart type pour determiner la précision voulue pour la gaussienne
elargissement: C'est l'élargissement ajouté à la fonction de comparaison
coeff_erosion: C'est le coefficient pour avoir une érosion moins violente
'''




import numpy as np
import cv2
import os
import sys
import time
from numba import jit

##Ficher des fonctions auxiliaires, placés dans un autre fichier pour améliorer la lisibilité

from Fonctions_auxiliaires import *
from Evaluation import *



#np.seterr(all = 'raise')
##Liste des dossiers (nom,nombre d'images) qui permet d'ajouter/supprimer facilement des bases de données

base = [("boulevard",2500), ("bungalows",1700), ("canoe",1189) ,("highway",1700) ,("parking",2500)]
base = [("highway",1700), ("parking",2500)]
base = [("office",2050),("sofa",2750),("PETS2006",1200)]

##Paramètres :

prm_conv     = 1
conv         = quick_convolution
avant        = True
apres        = False
filtre       = quick_erosion
n_filtre     = 3
kernel       = np.ones((n_filtre,n_filtre),np.float64)
mobile       = False
alpha        = 0.95
i_iteration  = 1
conv_tempo   = 1
prm_gauss    = 3
elargissement= 8
coeff_erosion= 0.6



#La fonction principale du code, Elle s'occupe de la comparaison et de l'envoie d'un résultat après comparaison des fonds.
#Caractéristiques : Les images sont traitées en noir en blanc uniquement

@jit
def Detecte(code_name):

    '''
    -code_name : Nom de la méthode utilisé, un dossier sera créé (le chemin est précisé pour mon PC mais peut être changé plus bas
    '''

    start = time.time()



    Occurences_tot = [0, 0, 0, 0, 0, 0, 0, 0] #Variable pour les Occurences (évaluation)
    ##Paramètre: Pour enregister les paramètres utilisés
    parametres = "code_name :"                 + str(code_name)     + "\n" +\
                 "prm_conv:"                   + str(prm_conv)      + "\n" +\
                 "conv:"                       + str(conv)          + "\n" +\
                 "convolution spaciale avant:" + str(avant)         + "\n" +\
                 "convolution spaciale après:" + str(apres)         + "\n" +\
                 "filtre:"                     + str(filtre)        + "\n" +\
                 "n_filtre"                    + str(n_filtre)      + "\n" +\
                 "conv_tempo:"                 + str(conv_tempo)    + "\n" +\
                 "prm_gauss"                   + str(prm_gauss)     + "\n" +\
                 "elargissement"               + str(elargissement) + "\n" +\
                 "coeff_erosion"               + str(coeff_erosion)

    ##Première boucle pour utiliser toutes les bases passées dans la liste "base"

    for elm in base :
        print(elm)
        Occurences = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        timer=time.time()
        timetable = np.array([0,0])
        ##Setup des chemins
        '''On crée  les chemins si ils manquent, avec les noms adaptés'''
        file,nbr = elm
        path_in  = r"E:\Users\Ahmed\Desktop\TIPE\BDD\\" + file + "\input"  #Chemin d'accès
        path_out = r"E:\Users\Ahmed\Desktop\TIPE\Results\\"+ code_name+"\\"+file #Chemin d'enregistrement
        path_gt  = r"E:\Users\Ahmed\Desktop\TIPE\BDD\\" + file + "\groundtruth"

        os.makedirs(path_out, exist_ok=True)

        ## --------------------------------------Le Code----------------------------------------
        # Ici, on met a jour nos grandeurs (moyenne) et on produit le rendu

        Fond = cv2.cvtColor(cv2.imread(path_in + "\img1.jpg"), cv2.COLOR_BGR2GRAY).astype(np.float64)  # Première image comme fond
        moyenne_cum, moyenne_carr = Moyenne_Sigma(Fond)  # On initialise la matrice de gaussiennes
        a, b = Fond.shape


        # Initialisation des matrices d'érosion:
        # Cela pour éviter de recréer des matrices à chaque boucle
        kernel1 = 255 * n_filtre * n_filtre * np.ones((a - n_filtre, b - n_filtre))
        kernel2 = 255 * n_filtre * n_filtre * np.ones((a - n_filtre,))
        kernel3 = 255 * n_filtre * n_filtre * np.ones((b - n_filtre,))

        ##Convolution Temporelle: Tentative"
        n = conv_tempo

        for i in range(2, nbr + 1):
            image = cv2.cvtColor(cv2.imread(path_in + "\img" + str(i) + ".jpg"), cv2.COLOR_BGR2GRAY).astype(np.float64)
            # Initialisation de la convolution (s'il ya)

            gt = cv2.cvtColor(cv2.imread(path_gt + "\gt" + str(i + n // 2) + ".png"), cv2.COLOR_BGR2GRAY)

            # Convolution

            if avant: image = conv(image, prm_conv)

            # Traitement/comparaison : utilisation de la fonction rendu()

            res = (rendu_v2(moyenne_cum, moyenne_carr, image,i, prm_gauss,elargissement))  # Image résultat en Booléens
            res2 = 255 * res  # Image résultat en Blanc et noir

            # Mise a jour de la moyenne de la gaussienne

            if mobile :
                moyenne_cum = Moyenne_mobile(image,moyenne,alpha)
            else:
                moyenne_cum += image

            moyenne_carr += image**2


            # Filtre morphologique

            if filtre:
                if filtre == quick_erosion:
                    for e in range(i_iteration):
                        res2 = filtre(res2, n_filtre, (kernel1, kernel2, kernel3),coeff_erosion)
                else:
                    for e in range(i_iteration):
                        res2 = filtre(res2, n_filtre, kernel)

            # Ecriture des résultats

            name = 'img_rendu' + str(i) + '.jpg'
            cv2.imwrite(path_out + "\\" + name, res2)

            ##------------------------------Partie d'évaluation---------------------------------------

            '''Regarder le module évaluation pour les détails'''
            Occurences = np.vstack((Occurences, np.append(evalue(res2, gt), i)))
            timetable = np.vstack((timetable, [time.time() - timer, i]))

        Occurences_tot += np.sum(Occurences[:, :8], axis = 0)
        np.savetxt(path_out + '\Data.txt', Occurences  , delimiter = ',')
        np.savetxt(path_out + '\ExeTime.txt', timetable, delimiter = ',')

        cv2.imwrite(path_out + "\Fond.jpg", moyenne_cum/i) # Image du fond

    #Sauvegarde des données pour une interprétation plus tard avec les fonctions du module Evaluation
    np.savetxt(r"E:\Users\Ahmed\Desktop\TIPE\Results\\" + code_name + '\Data.txt', Occurences_tot, delimiter = ',')
    f = open(r"E:\Users\Ahmed\Desktop\TIPE\Results\\" + code_name + "\\" + "Paramètres.txt", 'w')
    f.write(parametres)
    f.close()
    timetot = time.time() - start
    f = open(r"E:\Users\Ahmed\Desktop\TIPE\Results\\" + code_name + "\\" + "Time.txt", 'w')
    f.write(str(timetot))
    f.close()

    return ()
