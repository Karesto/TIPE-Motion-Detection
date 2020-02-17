########################### Codes d'évaluation ##########################

''' 
TP : True Positive : Détecté et devrait l'être             OK
FP : False Positive : Detecté et ne devrait pas l'être     NO
FN : False Negative : Non détecté et devrait l'être        NO
TN : True Negative : Non détecté et ne devrait pas l'être  OK

Colors used in the groundtruth
0: Black - Static
50: Dark Gray - Shadows
170: Light Gray - Unknown
255: White - Motion
'''

##-----------------------------------------------------Debut-----------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cv2
import sympy as sp

#Base de données utilisée
base = [("boulevard", 2500), ("bungalows", 1700), ("canoe", 1189), ("highway", 1700), ("parking", 2500)]
base = [("highway",1700),("parking",2500)]
base = [("office",2050),("sofa",2750),("PETS2006",1200)]

#Codes à comparer (Couleur est pour le plot, voir interprete V2)
Codes = [("Stat_elrg8")]
Couleur = ["Red"]


def evalue(resultat, groundtruth):
    '''

    Parameters
    ----------
    resultat : En entrée, la matrice résultante du code
    groundtruth : La matrice donnée par la base de donnée du rendu parfait

    Returns
    -------
    Occurences: Liste d'occurences des Erreurs
    Note: On utilise une opération assymétrique dans notre np.unique pour pouvoir briser la symétrie et distinguer les TP FP FN TN correctement

    '''

    # Pour une seule image, en compare les 2 ,renvoie le nb de TP FP FN TN
    Numbers, Occurences = np.unique(resultat.astype(np.int_) + 2 * groundtruth.astype(np.int_), return_counts=True)
    Occurences.resize(8)
    return (Occurences)

def interprete(Codes, shadow, name, save_path=""):
    '''
    Ceci est la première version de l'interpréteur, la deuxième version plus complète se trouve en bas


    Codes-str : liste des noms des codes qu'on va comparer
    shadow-booléen : si on considérera les ombres comme a detecter ou non - True : on veut, False: on veut pas
    save_path: Chemin de sauvegarde (a partir de TIPE/Resultts)
    '''

    path_out = r"E:\Users\Ahmed\Desktop\TIPE\Results\\"

    for elm, nbr in base:
        for code in Codes:
            Occurences = np.loadtxt(path_out + code + "\\" + elm + "\data.txt", delimiter=',')
            somme = np.sum(Occurences[1:, :8], axis=1)
            if shadow:

                plt.figure(1)  # True  Positive
                plt.plot(Occurences[1:, -1], (Occurences[1:, 7] + Occurences[1:, 4]) / somme)
                plt.figure(2)  # True  Negative
                plt.plot(Occurences[1:, -1], Occurences[1:, 0] / somme)
                plt.figure(3)  # False Positive
                plt.plot(Occurences[1:, -1], Occurences[1:, 5] / somme)
                plt.figure(4)  # False Negative
                plt.plot(Occurences[1:, -1], (Occurences[1:, 2] + Occurences[1:, 1]) / somme)
            else:
                plt.figure(1)  # True  Positive
                plt.plot(Occurences[1:, -1], Occurences[1:, 7] / somme)
                plt.figure(2)  # True  Negative
                plt.plot(Occurences[1:, -1], (Occurences[1:, 0] + Occurences[1:, 1]) / somme)
                plt.figure(3)  # False Positive
                plt.plot(Occurences[1:, -1], (Occurences[1:, 5] + Occurences[1:, 4]) / somme)
                plt.figure(4)  # False Negative
                plt.plot(Occurences[1:, -1], Occurences[1:, 2] / somme)

        os.makedirs(path_out + save_path + name + "\\" + elm , exist_ok=True)

        plt.figure(1)
        plt.xlabel('Frame')
        plt.ylabel('Pourcentage de Vrai positifs')
        plt.savefig(path_out + save_path + name + "\\" + elm + "\TP.png")
        plt.close()

        plt.figure(2)
        plt.xlabel('Frame')
        plt.ylabel('Pourcentage de Vrai Negatifs')
        plt.savefig(path_out + save_path + name + "\\" + elm + "\TN.png")
        plt.close()

        plt.figure(3)
        plt.xlabel('Frame')
        plt.ylabel('Pourcentage de Faux positifs')
        plt.savefig(path_out + save_path + name + "\\" + elm + "\FP.png")
        plt.close()

        plt.figure(4)
        plt.xlabel('Frame')
        plt.ylabel('Pourcentage de Faux Negatifs')
        plt.savefig(path_out + save_path + name + "\\" + elm + "\FN.png")
        plt.close()

        plt.close()

    return()

def interprete_v2(shadow, name, save_path=""):
    '''
    Parameters
    ----------

    shadow : Boolean : si on considérera les ombres comme a detecter ou non - True : on veut, False: on veut pas
    name : String : Nom du fichier où les sauvegarder
    save_path : String : Chemin de sauvegarde (a partir de TIPE/Results)

    Returns
    -------
    Enregistre dans le chemin spécifié des images qui représentent les grandeurs ci dessous et leur évolution par rapport au temps.
    On pourra ajouter plusieurs codes dans la liste au dessus pour les comparer
    Les grandeurs en question se trouvent sur www.changedetection.net
    '''

    path_out = r"E:\Users\Ahmed\Desktop\TIPE\Results\\"
    for elm in base:
        file, nbr = elm
        tempRoi  = np.loadtxt(r"E:\Users\Ahmed\Desktop\TIPE\BDD\\" + file + "\\temporalROI.txt")
        start,end = int(tempRoi[0]),int(tempRoi[1])

        for code,color in zip(Codes,Couleur):
            Occurences = np.loadtxt(path_out + code + "\\" + file + "\data.txt", delimiter=',')

            ExeTime = np.loadtxt(path_out + code + "\\" + file + '\ExeTime.txt', delimiter = ',')
            somme = np.sum(Occurences[1:, :8], axis=1)
            if shadow:
                # True  Positive
                TP = Occurences[start:end, 7] + Occurences[start:end, 4]
                # True  Negative
                TN = Occurences[start:end, 0]
                # False Positive
                FP = Occurences[start:end, 5]
                # False Negative
                FN = Occurences[start:end, 2] + Occurences[start:end, 1]
            else:
                # True  Positive
                TP = Occurences[start:end, 7]
                # True  Negative
                TN = Occurences[start:end, 0] + Occurences[start:end, 1]
                # False Positive
                FP = Occurences[start:end, 5] + Occurences[start:end, 4]
                # False Negative
                FN = Occurences[start:end, 2]



            plt.figure(1)  # Recall
            plt.style.use('ggplot')
            plt.plot(Occurences[start:end, -1], TP / (TP + FN), label = code, color = color)

            plt.figure(2)  # Specificity
            plt.style.use('ggplot')
            plt.plot(Occurences[start:end, -1], TN / (TN + FP), label = code, color = color)

            plt.figure(3)  # False Positive Rate
            plt.style.use('ggplot')
            plt.plot(Occurences[start:end, -1], FP / (FP + TN), label = code, color = color)

            plt.figure(4)  # False Negative Rate
            plt.style.use('ggplot')
            plt.plot(Occurences[start:end, -1], FN / (TP + FN), label = code, color = color)

            plt.figure(5)  # Percentage of Wrong Classifications
            txt = r"Pourcentage des fausses classifications:" + sp.latex((np.float16( 100 * np.sum(FN + FP) / np.sum((TP + FP + FN + TN))))) +"%"
            plt.style.use('ggplot')
            plt.plot(Occurences[start:end, -1], 100 * (FN + FP) / (TP + FN + FP + TN), label = code, color = color)
            plt.figure(5).text( 0.5, -0.1, sp.latex(txt),ha = 'center',fontsize=16)

            plt.figure(6)  # Precision
            plt.style.use('ggplot')
            plt.plot(Occurences[start:end, -1],np.divide(TP,(TP+FP), out=np.zeros_like(TP), where=(TP+FP)!=0), label = code, color = color)

            plt.figure(7)  # Temps d'execution
            # Ecriture de la vitesse moyenne et du temps de réponse moyen sur le graphe
            txt = r"$V_{moy} =$" + sp.latex((np.float16(ExeTime[-1,1]/ExeTime[-1,0]))) + r" $frame/s$" + "\n" +\
                  r"$T_{réponse} =$"  + sp.latex((np.float16(ExeTime[-1,0]/ExeTime[-1,1]))) + r" $seconde/frame$"
            plt.style.use('ggplot')
            plt.plot(ExeTime[:,1],ExeTime[:,0], label = code, color = color)
            plt.figure(7).text( 0.5, -0.1, sp.latex(txt),ha = 'center',fontsize=16)

        os.makedirs(path_out + save_path + name + "\\" + file , exist_ok=True)

        plt.figure(1)
        plt.xlabel('Frame')
        plt.ylabel('Recall')
        plt.savefig(path_out + save_path + name + "\\" + file + "\Recall.png")
        plt.close()

        plt.figure(2)
        plt.xlabel('Frame')
        plt.ylabel('Specifity')
        plt.savefig(path_out + save_path + name + "\\" + file + "\Specifity.png")
        plt.close()

        plt.figure(3)
        plt.xlabel('Frame')
        plt.ylabel('False Positive Rate')
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.savefig(path_out + save_path + name + "\\" + file + "\False Positive Rate.png")
        plt.close()

        plt.figure(4)
        plt.xlabel('Frame')
        plt.ylabel('False Negative Rate')
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.savefig(path_out + save_path + name + "\\" + file + "\False Negative Rate.png")
        plt.close()

        plt.figure(5)
        plt.xlabel('Frame')
        plt.ylabel('Percentage of Wrong Classifications')
        axes = plt.gca()
        axes.set_ylim([0, 100])
        plt.savefig(path_out + save_path + name + "\\" + file + "\Percentage of Wrong Classifications.png",transparent = True,bbox_inches='tight',pad_inches =0.5)
        plt.close()

        plt.figure(6)
        plt.xlabel('Frame')
        plt.ylabel('Precision')
        plt.savefig(path_out + save_path + name + "\\" + file + "\Precision.png")
        plt.close()

        plt.figure(7)
        plt.xlabel('Frame')
        plt.ylabel('Execution Time in seconds')
        plt.style.use('ggplot')
        plt.savefig(path_out + save_path + name + "\\" + file + "\Exectime.png",transparent = True,bbox_inches='tight',pad_inches =0.5,)
        plt.close()
        plt.close()

def heatmap_var(Code_name):
    # A noter que je trace une carte de chaleur de la variance de chaque pixel par rapport au temps

    path_out = r"E:\Users\Ahmed\Desktop\TIPE\Results\\"

    for file,nbr in base: #On en fait une pour chaque élément
        path_in = r"E:\Users\Ahmed\Desktop\TIPE\Results\\" + Code_name + "\\" + file  # Chemin d'accès
        print(path_in)
        tab_variance = cv2.cvtColor(cv2.imread(path_in + "\sigma.jpg"), cv2.COLOR_BGR2GRAY)
        sns.set()

        plotte = sns.heatmap(tab_variance,cmap=sns.cubehelix_palette(8)) #Tracé de la carte de chaleur
        plt.xticks([])
        plt.yticks([])
        plt.savefig(r"E:\Users\Ahmed\Desktop\TIPE\Results\\" + Code_name + "\\" + file + ".png",transparent = True)
        plt.style.use('ggplot')
        plt.show()
        plt.close()

    return()

def Histogramme(pixels):
    '''

    Parameters
    ----------
    pixels : Liste des pixels où calculer en compte par rapport a chaque base de données

    Returns
    -------
    Enregistre dans le chemin spécifié (le dossier results ici) les différents histogrammes des pixels

    '''
    for first,pixel in zip(base,pixels): #On en fait une pour chaque élément
        file,nbr = first
        valeurs = []
        a,b = pixel[0],pixel[1]
        path_in = r"E:\Users\Ahmed\Desktop\TIPE\BDD\\" + file + "\input"  # Chemin d'accès
        for i in range (1,nbr): #Création des tableaux de variance
            valeurs.append(cv2.cvtColor(cv2.imread(path_in + "\img" + str(i) + ".jpg"), cv2.COLOR_BGR2GRAY)[a,b])

        plt.style.use('seaborn')

        #Noms des axes
        plt.xlabel('Intensité')
        plt.ylabel("Nombre d'occurences dans la vidéo ")
        #Valeur de l'écart type et insertion dans le graphe
        txt = r"$\sigma = $" + str(np.float16(np.std(valeurs)))
        plotte = sns.distplot(valeurs,kde = False)
        #Tracé sans fonction de répartition
        ax= sns.distplot(valeurs)
        ax.text(0.5, -.17, txt, ha='center',transform=ax.transAxes, fontsize = 18)
        plt.savefig(r"E:\Users\Ahmed\Desktop\TIPE\Results\\" + file + str(a) +"_"+ str(b) + "w.png",transparent = True,bbox_inches='tight',pad_inches =0.5)
        plt.close()


        # Tracé avec fonction de répartition
        plt.xlabel('Intensité')
        plt.ylabel("Nombre d'occurences dans la vidéo (normalisé) ")
        ax= sns.distplot(valeurs)
        ax2 = ax.twinx()
        sns.boxplot(x = valeurs, ax = ax2)
        ax2.set(ylim = (-.5, 10))
        ax.text(0.5, -.17, txt, ha='center',transform=ax.transAxes,fontsize = 18)
        plt.savefig(r"E:\Users\Ahmed\Desktop\TIPE\Results\\" + file + str(a) + "_" + str(b) + ".png",transparent = True,bbox_inches='tight',pad_inches =0.5)

        plt.close()

    return()


















