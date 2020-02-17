from numba import jit
import numpy as np

@jit
def BB(a):
    def boulon(a):
        return("bidule")
    L= [a]
    for i in range(a):
        L = L
        print(L)
    def karma(sina):
        return(sina+1)
    return(karma(L[0]))

@jit
def sommation(img):
    a,b = img.shape
    fonc = lambda x: 3
    res = img.astype(np.float32)
    for i in range(a - 2, -1, -1):
        res[i, b - 1] += res[i + 1, b - 1]
    for j in range(b - 2, -1, -1):
        res[a - 1, j] += res[a - 1, j + 1]
    for i in range(a - 2, -1, -1):
        for j in range(b - 2, -1, -1):
            res[i, j] += res[i + 1, j] + res[i, j + 1] - res[i + 1, j + 1]
    return (res,fonc(a))

@jit
def test(N):
    b = np.random.random((N,N))
    # t0 = time()
    A = sommation(b)
    # t1 = time()
    # numpysommation(b)
    # t2 = time()
    # result = "Pour une matrice en entrée de taille {},\n la sommation sans numpy a pris {}s\n celle avec numpy a pris {}s".format(N,t1-t0, t2-t1)
    print(A)