import numpy as np

# On suppose qu'on a note matrice condition Cond[t,x,y], on peut ajouter z mais ce sera compliqué a comprndre avec numpy
# On ne regarde que selon x et y (l'altitude n'est pas importante
# On définit N la taille du tableau en x, y en variable globale

Cond =[]

N = 10

a.reshape(x.shape[0],-1).argmax(1)
# tenta : (sinon on le fait avec des modulos, pas sur que le axis marche mais bon ^^)
ind = numpy.unravel_index(np.argmax(cond,axis=(1,2)), (N,N))


#si ça ne marche pas la solution lente.
indices = []
for i in range (len(cond[:,0,0])):
    indices.append(np.argmax[cond[]])

In [263]: x = np.random.randint(10, size=(4,3,3))

In [264]: x
Out[264]:
array([[[0, 9, 2],
        [7, 7, 8],
        [2, 5, 9]],

       [[1, 7, 2],
        [8, 9, 0],
        [2, 8, 3]],

       [[7, 5, 0],
        [7, 1, 6],
        [5, 1, 1]],

       [[0, 7, 3],
        [5, 4, 1],
        [9, 8, 9]]])

In [265]: idx = x.reshape(x.shape[0],-1).argmax(-1)

In [266]: np.unravel_index(idx, x.shape[-2:])
Out[266]: (array([0, 1, 0, 2]), array([1, 1, 0, 0]))