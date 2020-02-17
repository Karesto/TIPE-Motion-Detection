########################### Codes Auxiliaires ##########################

#Les fonctions n'acceptent que des images monodimentionelles


import numpy as np
import cv2
from numba import jit

##Fonctions de base pour les Gaussiennes:
@jit
def Moyenne_Sigma(image):
    '''
    Initialisation de la matrice de gaussiennes
    Image: Matrice de l'image, ne sert qu'a prendre les dimensions.
    Initialisation pauvre
    '''
    return (np.copy(image),np.copy(image)**2)

@jit
def rendu(moyenne, sigma, image, C=1,elargissement = 0):
    '''
    Renvoie le rendu de la comapraison pour le modèle probabilsite des gaussiennes passées en paramètre

    Parameters
    ----------
    moyenne : 2D-Array
              Moyenne  pour chaque pixel

    sigma   : 2D-Array
              Ecart-type  pour chaque pixel
    image   : 2D-Array
              Image a taiter

    C       : Float
              Caractérise la précision du modèle

    elargissement: Float
                   Décalage écart-type

    Returns
    -------
        Bool 2D-Array: Faux si considéré comme immobile, Vrai si considéré comme mobile
    '''

    return ((np.abs(moyenne - image) >= (C * sigma + elargissement)).astype(np.float64))

@jit
def rendu_v2(moyenne_cum,moyenne_carr,image,i, C=1, elargissement = 0):
    '''

    Parameters
    ----------
    moyenne_cum : 2D-Array
                  Valeurs cumulées des pixels

    moyenne_carr : 2D-Array
                  Valeurs cumulées du carré des pixels

    image   : 2D-Array
              Image a taiter
    i: int
       le nombre d'itération

    C       : Float
              Caractérise la précision du modèle

    elargissement: Float
                   Décalage écart-type
    Returns
    -------

    '''
    moyenne = moyenne_cum/i
    sigma = np.sqrt(moyenne_carr/i-(moyenne)**2)
  #  print(sigma[0,0])
    return(rendu(moyenne,sigma,image,C,elargissement))

@jit
def rendu_diff(image,moyenne,seuil):
    return ((np.abs(moyenne - image) >= seuil).astype(np.uint8))

@jit
def rendu_diff_succ(image1,image2,seuil):
    return ((np.abs(image1- image2) >= seuil).astype(np.uint8))

@jit
def sommation(img: "2D Array"):
    '''
    Sommation(img) renvoie une image res telle que res[i][j] est la somme de la sous matrice img[i:][j:]

    Parameters
    ----------
    img: image a sommer

    Returns
    -------
    res where res[i][j] = np.sum(img[i:][j:])
    '''
    a, b = np.shape(img)
    res = img.astype(np.float)
    # Le astype sert pour ne pas avoir du module 255

    for i in range(a - 2, -1, -1):
        res[i, b - 1] += res[i + 1, b - 1]
    for j in range(b - 2, -1, -1):
        res[a - 1, j] += res[a - 1, j + 1]
    for i in range(a - 2, -1, -1):
        for j in range(b - 2, -1, -1):
            res[i, j] += res[i + 1, j] + res[i, j + 1] - res[i + 1, j + 1]
    return (res)

def numpysommation(img):
    '''
    numpysommation(img) renvoie une image res telle que res[i][j] est la somme de la sous matrice img[i:][j:]

    Parameters
    ----------
    img: image a sommer

    Returns
    -------
    res where res[i][j] = np.sum(img[i:][j:])

    Notes
    -----
    This works by first making a res that contains the sum of only the columns using cumsum and then doing the same operation
    on the lines, on penserait que ce serait plus lent que sommation normale

    - trouver une meilleur solution (moins d'inversions ?)

    '''

    a, b = np.shape(img)
    res = img.astype(np.float)
    res = res[:,::-1] # on inverse les colonnes
    res = res.cumsum(axis = 1) # on somme sur les lignes
    res = res[::-1,::-1] # on réinverse les colonnes, on inverse les lignes
    res = res.cumsum(axis = 0) # on somme sur les colonnes
    res = res[::-1] # on revient à l'ordre normal des lignes
    return res

#Moyennes et Ecart-types

#Outil de mise à jour peu optimisé (niveau erreurs de calcul)
def Moyenne_Arith (img, moyenne,n):
    '''
            Renvoie la moyenne arithmétique à t = n partir de la moyenne à t = n-1

    Parameters
    ----------
    img     :   2D Numpy Array
                Image à t = n+1

    Moyenne :   2D Numpy Array
                Moyenne(t = n)

    n       : Int
              Le nombre d'itérations
    Returns
    -------
    2D Numpy Array : Moyenne arithmétique à t = n+1

    '''

    return (     ( (n-1) * moyenne + img) /n )

#Outil de mise à jour peu optimisé (niveau erreurs de calcul)
def Ecart_type_Arith (img, sigma, moyenne, moyenne_new, n):
    '''
        Renvoie l'écart-type arithmétique à t+1 partir de la moyenne à t  en utilisant la formule de Koenig-Huggens

    Parameters
    ----------
    img         : 2D Numpy Array
                  Image a t

    sigma       : 2D Numpy Array
                  Ecart-type(t

    moyenne     : 2D Numpy Array
                  Moyenne(t)

    moyenne_new : 2D Numpy Array
                  Moyenne(t+1)

    n : Int
        Le nombre d'itérations

    Returns
    -------
    2D Numpy Array : Ecart-Type arithmétique à t+1

    '''
    return( np.sqrt( ((sigma**2 + moyenne**2)*(n-1) +img**2)/(n) - moyenne_new**2 )  )

def Moyenne_mobile(img,moyenne,alpha):
    '''
            Renvoie un moyenne mobile (exponentielle) a t = n a partie de la moyenne a t = n-1
    Parameters
    ----------
    img     :   2D Numpy Array
                Image à t = n+1

    Moyenne :   2D Numpy Array
                Moyenne(t = n)

    alpha   : Float
              Facteur d'apprentissage
    Returns
    -------

    '''

    return( (1-alpha)*moyenne + alpha*img )

def Ecart_type_mobile(img,sigma,moyenne,alpha):
    '''
        Renvoie l'écart-type mobile à t partir de la moyenne à  t+1

    Parameters
    ----------
    img         : 2D Numpy Array
                  Image a t

    sigma       : 2D Numpy Array
                  Ecart-type(t)

    moyenne     : 2D Numpy Array
                  Moyenne(t+1)

    alpha       : Float
                  Facteur d'apprentissage

    Returns
    -------
    2D Numpy Array : Ecart-Type arithmétique à t = n+1
    '''

    return(np.sqrt( (1-alpha)*sigma**2 + alpha*(img-moyenne)**2 )  )




##Convolutions
@jit
def conv1(img, prm, n=3):
    '''
        Effectue une convolution sur l'image passée en paramètre (Ne modifie pas l'image)
        La convolution accorde prm comme poids au pixel principal et un poids qui s'affaiblit plus on s'éloigne
        Ne s'effectue pas sur les bords

    Parameters
    ----------
    img : 2D-Array
          Image à convoluer

    prm : Float (between 0 and 1)
          Paramètre de la convolution

    n   : Int
          Taille de la convolution
          Doit être impair
          Ne fonctionne pas sur cette fonction

    Returns
    -------
    res: Convoluted Image according to the parametres

    '''
    a, b = img.shape
    d = n // 2
    res = img.astype(np.float)
    prm1 = (1 - prm) / 6
    prm2 = (1 - prm) / 12
    res[1:a - d, d:b - 1] = prm * img[1:a - 1, 1:b - 1] + (
            prm1 * img[0:a - 2, 1:b - 1] + prm1 * img[2:a, 1:b - 1] + prm1 * img[1:a - 1, 0:b - 2] + prm1 * img[
                                                                                                            1:a - 1,
                                                                                                            2:b]) + (
                                    prm2 * img[2:a, 0:b - 2] + prm2 * img[0:a - 2, 0:b - 2] + prm2 * img[0:a - 2,
                                                                                                     2:b] + prm2 * img[
                                                                                                                   2:a,
                                                                                                                   2:b])
    return (res)

@jit
def conv2(img: "2D Array", prm: "float between 0 and 1", n: "do not care about this one" = 3):
    '''
        Effectue une convolution sur l'image passée en paramètre (Ne modifie pas l'image)
        La convolution accorde prm comme poids au pixel principal et le même autre aux pixels environnants
        Ne s'effectue pas sur les bords

    Parameters
    ----------
    img : 2D-Array
          Image à convoluer

    prm : Float (between 0 and 1)
          Paramètre de la convolution

    n   : int
          Taille de la convolution
          Doit être impair
          Ne fonctionne pas sur cette fonction

    Returns
    -------
    res: Convoluted Image according to the parametres

    '''
    a, b = img.shape
    res = img.astype(np.float)

    prm1 = (1-prm) / 8
    res[1:a-1, 1:b-1] = prm * img[1:a-1, 1:b-1]+(prm1 * img[0:a-2, 1:b-1]+prm1 * img[2:a, 1:b-1]+prm1 * img[1:a-1, 0:b-2]+prm1 * img[1:a-1, 2:b])+(prm1 * img[2:a, 0:b-2]+prm1 * img[0:a-2, 0:b-2]+prm1 * img[0:a-2, 2:b]+prm1 * img[2:a, 2:b])
    return (res.astype(np.uint8))

@jit
def conv2_generalisee(img: "2D Array", prm: "float between 0 and 1", n: "Size of the Operation" = 3):
    '''
        Effectue une convolution sur l'image passée en paramètre (Ne modifie pas l'image)
        La convolution accorde prm comme poids au pixel principal et le même autre aux pixels environnants
        Ne s'effectue pas sur les bords
    Parameters
    ----------
    img : 2D-Array
          Image à convoluer

    prm : Float (between 0 and 1)
          Paramètre de la convolution

    n   : int
          Taille de la convolution
          Doit être impair

    Returns
    -------
    res: Convoluted Image according to the parametres

    '''
    sum = sommation(img)
    a, b = np.shape(img)
    res = img.astype(np.float)
    d = n // 2
    landa  = (1 - prm) / (n * n -1)
    landa2 = (-(1 -  prm) / (n * n - 1) + prm)
    for i in range(d, a - d - 1):
        for j in range(d, b - d - 1):
            res[i, j] = (sum[i - d, j - d] - sum[i + d + 1, j - d] - sum[i - d, j + d + 1] + sum[i + d + 1, j + d + 1]) * landa + landa2 * img[i, j]
    for i in range(d, a - d - 1):
        res[i, b - d - 1] = (sum[i - d, b - n] - sum[i + d + 1, b - n]) * landa + landa2 * img[i, b - d - 1]
    for j in range(d, b - d - 1):
        res[a - d - 1, j] = (sum[a - n, j - d] - sum[a - n, j + d + 1]) * landa + landa2 * img[a - d - 1, j]
    res[a - 1 - d, b - 1 - d] = (sum[a - n, b - n]) * landa + landa2 * img[a - 1 - d, b - 1 - d]
    return (res)

@jit
def quick_convolution(img: "2D Array", prm: "float between 0 and 1", n: "Size of the Operation" = 3):
    '''
        Effectue une convolution sur l'image passée en paramètre (Ne modifie pas l'image)
        La convolution accorde prm comme poids au pixel principal et le même autre aux pixels environnants
        Ne s'effectue pas sur les bords
    Parameters
    ----------
    img : 2D-Array
          Image à convoluer

    prm : Float (between 0 and 1)
          Paramètre de la convolution

    n   : int
          Taille de la convolution
          Doit être impair

    Returns
    -------
    res: Convoluted Image according to the parametres

    '''
    sum = sommation(img)
    a, b = np.shape(img)
    res = img.astype(np.float64)
    d = n // 2
    landa = (1 - prm) / (n * n - 1)
    landa2 = (-(1 - prm) / (n * n - 1) + prm)
    res[d:a - d - 1, d:b - d - 1] = ((sum[:a - n, :b - n] - sum[:a - n, n:b] - sum[n:a, :b - n] + sum[n:a, n:b]) * landa + landa2 * img[d:a - d - 1, d:b - d - 1])
    # Avant dernière colonne
    res[d:a - d - 1, b - d - 1] = (
            (sum[:a - n, b - n] - sum[n:a, b - n]) * landa + landa2 * img[d:a - d - 1, b - d - 1])
    # Avant dernière ligne
    res[a -  d - 1, d:b - d - 1] = (sum[a - n, :b - n] - sum[a - n, n:b]) * landa + landa2 * img[a - d - 1, d:b - d - 1]
    # Point du coin
    res[a - 1 - d, b - 1 - d] = (sum[a - n, b - n]) * landa + landa2 * img[a - 1 - d, b - 1 - d]
    return (np.ceil(res.astype(np.float64)))


##Filtres Morphologiques
@jit
def erosion(img, n, kernel=None):



    '''
        Effectue une érosion sur une image (filtre morphologique), en noir et blanc uniquement

    Paramètres
    ----------

        img : 2D-Array
              Image a éroder
        n   : int
              Taille de l'érosion
        kernel : Kernel Matrix
                 kernel = 255 * np.ones((n, n))
                 Paramètre non nécéssaire
    Returns
    -------
        res : 2D-Array  after the operation


    '''

    ''' n impair uniquement , taille du kernel de travail'''
    ''' kernel : matrice d'érosion -->   kernel = 255*np.ones((n,n),np.uint8) il est inclu car ça diminue la complexité de le réinclure '''

    a, b = np.shape(img)
    d = n // 2
    kernel = 255 * np.ones((n, n))
    res = np.copy(img)
    for i in range(d, a - d):
        for j in range(d, b - d):
            res[i, j] = 255*np.array_equal(img[i - d:i + d + 1, j - d:j + d + 1], kernel)
    return (res)

@jit
def quick_erosion(img: "2D Array", n: "Size of the Operation", kernel: "Read docstring for format", coeff: "float between 0 and 1" = 1) -> object:
    '''
    Effectue une érosion sur une image (filtre morphologique), en noir et blanc uniquement
    La convolution accorde prm comme poids au pixel principal et le même autre aux pixels environnants
    Ne s'effectue pas sur les bords

    Paramètres
    ----------

        img : 2D-Array
              Image a éroder
        n   : int
              Taille de l'érosion
              Doit être impair
        kernel : 3-Tuple of matrix
                (kernel1,kernel2,kernel3)
                kernel1 = 255 * n * n * np.ones((a - n, b - n))
                kernel2 = 255 * n * n * np.ones((a - n,))
                kernel3 = 255 * n * n * np.ones((b - n,))
                Avec a,b = img.shape .On les passe en argument pour éviter qu'ils soient recrées a chaque passage (dans la boucle du code)
        Coeff : Float (between 0 and 1)
                Coeffecient pour diluer l'érosion, en phase de test, entre 0 (non) et 1 (pour une éro normale)
    Returns
    -------

        res : 2D-Array  after the operation


    '''
    sum = sommation(img)
    a, b = np.shape(img)
    res = img.astype(np.float)
    d = n // 2
    kernel1, kernel2, kernel3 = kernel
    res[d:a - d - 1, d:b - d - 1] = 255*(
        np.greater_equal((sum[:a - n, :b - n] - sum[:a - n, n:b] - sum[n:a, :b - n] + sum[n:a, n:b]), coeff*kernel1))
    # Avant dernière colonne
    res[d:a - d - 1, b - d - 1] = 255*(np.greater_equal(sum[:a - n, b - n] - sum[n:a, b - n], coeff*kernel2))
    # Avant dernière ligne
    res[a - d - 1, d:b - d - 1] = 255*(np.greater_equal(sum[a - n, :b - n] - sum[a - n, n:b], coeff*kernel3))
    # Point du coinerosion
    res[a - 1 - d, b - 1 - d] = 255*((sum[a - n, b - n]) >= coeff * n * n)
    return (res)

@jit
def quick_dilatation(img: "2D Array", n: "Size of the Operation", kernel: "Read docstring for format"):

    '''
        Effectue une dilatation sur une image (filtre morphologique), en noir et blanc uniquement

    Paramètres
    ----------

        img : 2D-Array
              Image a dilater
        n   : int
              Taille de la dilatation
              Doit être impair
        kernel : 3-Tuple of matrix
                (kernel1,kernel2,kernel3)
                kernel1 = np.zeros((a - n, b - n))
                kernel2 = np.zeros((a - n,))
                kernel3 = np.zeros((b - n,))
                Avec a,b = img.shape .On les passe en argument pour éviter qu'ils soient recrées a chaque passage (dans la boucle du code)

    Returns
    -------

         res : 2D-Array  after the operation

    '''

    sum = sommation(img)
    a, b = np.shape(img)
    res = img.astype(np.float)
    d = n // 2
    # Ils sont passés en argument pour ne pas avoir a les recréer a chaque fois
    kernel1, kernel2, kernel3 = kernel

    res[d:a - d - 1, d:b - d - 1] = (
        np.not_equal((sum[:a - n, :b - n] - sum[:a - n, n:b] - sum[n:a, :b - n] + sum[n:a, n:b]), kernel1))
    # Avant dernière colonne
    res[d:a - d - 1, b - d - 1] = (np.not_equal(sum[:a - n, b - n] - sum[n:a, b - n], kernel2))
    # Avant dernière ligne
    res[a - d - 1, d:b - d - 1] = (np.not_equal(sum[a - n, :b - n] - sum[a - n, n:b], kernel3))
    # Point du coinerosion
    res[a - 1 - d, b - 1 - d] = ((sum[a - n, b - n]) != 0)
    return (res * 255)

@jit
def quick_dilation_avec_coeff(img: "2D Array", n: "Size of the Operation", kernel: "Read docstring for format" , coeff: "float between 0 and 1" = 0):

    '''
        Effectue une dilatation sur une image (filtre morphologique), en noir et blanc uniquement

    Paramètres
    ----------

        img : 2D-Array
              Image a dilater
        n   : int
              Taille de la dilatation
              Doit être impair
        kernel : 3-Tuple of matrix
                (kernel1,kernel2,kernel3)
                kernel1 = 255 * n * n * np.ones((a - n, b - n))
                kernel2 = 255 * n * n * np.ones((a - n,))
                kernel3 = 255 * n * n * np.ones((b - n,))
                Avec a,b = img.shape .On les passe en argument pour éviter qu'ils soient recrées a chaque passage (dans la boucle du code)
        Coeff : Float (between 0 and 1)
                Coeffecient pour diluer l'érosion, en phase de test, entre 0 (non) et 1 (pour une éro normale)
    Returns
    -------

         res : 2D-Array  after the operation

    '''

    sum = sommation(img)
    a, b = np.shape(img)
    res = img.astype(np.float)
    d = n // 2
    # Ils sont passés en argument pour ne pas avoir a les recréer a chaque fois
    kernel1, kernel2, kernel3 = kernel

    res[d:a - d - 1, d:b - d - 1] = (
        np.greater_equal((sum[:a - n, :b - n] - sum[:a - n, n:b] - sum[n:a, :b - n] + sum[n:a, n:b]), coeff*kernel1))
    # Avant dernière colonne
    res[d:a - d - 1, b - d - 1] = (np.greater_equal(sum[:a - n, b - n] - sum[n:a, b - n], coeff*kernel2))
    # Avant dernière ligne
    res[a - d - 1, d:b - d - 1] = (np.greater_equal(sum[a - n, :b - n] - sum[a - n, n:b], coeff*kernel3))
    # Point du coin
    res[a - 1 - d, b - 1 - d] = ((sum[a - n, b - n]) >= coeff * n * n)
    return (res * 255)

