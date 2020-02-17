import numpy as np
from numba import jit
from time import time


@jit
def sommation(img):
    '''
    sommation(img) renvoie une image res telle que res[i][j] est la somme de la sous matrice img[i:a][j:b]
    img: image a sommer
    '''

    a, b = img.shape
    res = img.astype(np.float32)
    # Le astype sert pour ne pas avoir du module 255
    def boo(x):
        return 3
    for i in range(a - 2, -1, -1):
        res[i, b - 1] += res[i + 1, b - 1]
    for j in range(b - 2, -1, -1):
        res[a - 1, j] += res[a - 1, j + 1]
    for i in range(a - 2, -1, -1):
        for j in range(b - 2, -1, -1):
            res[i, j] += res[i + 1, j] + res[i, j + 1] - res[i + 1, j + 1]
    return (res,boo(img))


def numpysommation(img):
    '''
    Travail : 
    - essayer de se convaincre pourquoi ça marche.
    - trouver une meilleur solution (moins d'inversions ?)
    '''
    res = img.astype(np.float)
    res = res[:,::-1] # on inverse les colonnes
    res = res.cumsum(axis = 1) # on somme sur les lignes
    res = res[::-1,::-1] # on réinverse les colonnes, on inverse les lignes
    res = res.cumsum(axis = 0) # on somme sur les colonnes
    res = res[::-1] # on revient à l'ordre normal des lignes
    return res
    

def test(N):
    b = np.random.random((N,N))
    t0 = time()
    sommation(b)
    t1 = time()
    numpysommation(b)
    t2 = time()
    result = "Pour une matrice en entrée de taille {},\n la sommation sans numpy a pris {}s\n celle avec numpy a pris {}s".format(N,t1-t0, t2-t1)
    print(result)

#>>> test(1000)
# Pour une matrice en entrée de taille 1000,
#  la sommation sans numpy a pris 1.3101692199707031s
#  celle avec numpy a pris 0.02482318878173828s
