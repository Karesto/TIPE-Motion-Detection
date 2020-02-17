########################################---Code Principal---########################################
'''
-prm_conv   : Réel entre 0 et 1 Paramète de la convolution (pourra être dédoublé pour la convolution avant/convolution après)
-Conv       : Fonction , C'est la fonction de convolution a utiliser, on utilisera uniquement une seule pour les deux convolutions avant et après, fonctions disponible : conv1, conv2
-avant      : Booléen pour déterminer si on utilise la convolution sur les images avant de les comparer
-après      : Booléen pour déterminer si on utilise la convolution sur les images après les avoir comparé
-filtre     : Booléen pour déterminer si on utilise un filtre morphologique (auquel cas il est inséré a la place du booléen)
-n_filtre   : Taille du Noyau du filtre
-kernel     : Matrice pour le filtre
-i_iteration: Nb d'application du filtre morphologique
Fond_prm    : Paramètre poru déterminer si on utile un fond
maj_fond    : Détermine quelle mise à jour de fond faire
prm_adap    : Paramètre alpha de la moyenne mobile
seuil       : Seuil de comparaison lors d'une différence
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

##Liste des dossiers (nom,nombre d'images) qui permet d'ajouter/supprimer facilement des bases de données

base = [("highway",1700),("parking",2500)]
base = [("office",2050),("sofa",2750),("PETS2006",1200)]

##Paramètres :


prm_conv    = 1
conv        = False
avant       = False
apres       = False
filtre      = quick_erosion
n_filtre    = 3
kernel      = np.ones((n_filtre,n_filtre),np.float64)
i_iteration = 0
Fond_prm    = True
maj_fond    = "mobile"
prm_adap    = 0.96
seuil       = 55




#La fonction principale du code, Elle s'occupe de la comparaison et de l'envoie d'un résultat après comparaison des fonds.
#Caractéristiques : Les images sont traitées en noir en blanc uniquement


@jit
def Detecte(code_name):

    '''
    -code_name : str : Nom de la méthode utilisé, un dossier sera créé (le chemin est précisé pour mon PC mais peut être changé plus bas
    '''

    start = time.time()

    Occurences_tot = [0, 0, 0, 0, 0, 0, 0, 0] #Variable pour les Occurences (évaluation)
    ##Paramètre: Pour sauver les paramètres utilisés
    parametres = "code_name :"                 + str(code_name)  + "\n" +\
                 "prm_conv:"                   + str(prm_conv)   + "\n" +\
                 "conv:"                       + str(conv)       + "\n" +\
                 "convolution spatiale avant:" + str(avant)      + "\n" +\
                 "convolution spatiale après:" + str(apres)      + "\n" +\
                 "filtre:"                     + str(filtre)     + "\n" +\
                 "n_filtre"                    + str(n_filtre)   + "\n" +\
                 "Fond_prm"                    + str(Fond_prm)   + "\n" +\
                 "maj_fond"                    + str(maj_fond)   + "\n" +\
                 "prm_adap"                    + str(prm_adap)   + "\n" +\
                 "seuil"                       + str(seuil)      + "\n"


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



        Fond = cv2.cvtColor(cv2.imread(path_in + "\img1.jpg"), cv2.COLOR_BGR2GRAY)  # Première image comme fond
        a, b = Fond.shape
        moyenne_cum = np.copy(Fond).astype(np.uint32)

        # Initialisation des matrices d'érosion:

        kernel1 = 255 * n_filtre * n_filtre * np.ones((a - n_filtre, b - n_filtre))
        kernel2 = 255 * n_filtre * n_filtre * np.ones((a - n_filtre,))
        kernel3 = 255 * n_filtre * n_filtre * np.ones((b - n_filtre,))
        image = Fond

        for i in range(2, nbr + 1):

            img_old = image

            image = cv2.cvtColor(cv2.imread(path_in + "\img" + str(i) + ".jpg"), cv2.COLOR_BGR2GRAY)
            gt = cv2.cvtColor(cv2.imread(path_gt + "\gt" + str(i) + ".png"), cv2.COLOR_BGR2GRAY)

            # Convolution

            if avant: image = conv(image, prm_conv)

            # Traitement/comparaison :
            if Fond_prm :
                res = rendu_diff(image, Fond, seuil)  # Image résultat en Booléens
                res2 = 255 * res  # Image résultat en Blanc et noir
            else:
                res = rendu_diff_succ(image,img_old, seuil)  # Image résultat en Booléens
                res2 = 255 * res  # Image résultat en Blanc et noir

            if maj_fond == "mobile" :
                Fond = Moyenne_mobile(image,Fond,prm_adap)
            elif maj_fond == "arith":
                moyenne_cum += image
                Fond = moyenne_cum/i

            # Convolution

            if apres: res = conv(res, prm_conv)

            # Filtre morphologique

            if filtre:
                if filtre == quick_erosion:
                    for e in range(i_iteration):
                        res2 = filtre(res2, n_filtre, (kernel1, kernel2, kernel3))
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
        cv2.imwrite(path_out + "\Fond.jpg", Fond)

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
