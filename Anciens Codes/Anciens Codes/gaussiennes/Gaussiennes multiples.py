import numpy as np 
import scipy.stats as stats


''' Arguments :
x : Valeur de notre pixel
mu : Moyenne
sig: Ecart-type
tab: liste contenue dans notre matrice, rangée comme ceci : [Poids1, moyenne1, ec1, moyenne2, ec2]
N:Nombre d'éléments dans notre liste de données

'''





def P(x,mu,sig):                  #P(x|x dans Ci)
    t = (x-mu)/sig
    res = stats.norm.pdf(t) /sig
    return(res)
    
def Prob (x,tab):
    w1, mu1, sig1, mu2, sig2 = tab
    p_aux = 1 + ((1-w1)*P(x,mu2,sig2))/(w1*P(x,mu1,sig1))
    return(1/p_aux)
    
def wnew(w,N,prob):    #Nouveau poid
    return((w1*N+prob)/(N+1))
    
def signew (x,tab,N):                       # Renvoie deux valeurs sigma1 et sigma 2 en mise a jour
    w1, mu1, sig1, mu2, sig2 = tab
    p1 = Prob(x,tab)
    new1 = np.sqrt((sig1**2*w1*N+p1*(x-mu1)**2)/(w1*N+p1))
    new2 = np.sqrt((sig2**2*(1-w1)*N+(1-p1)*(x-mu2)**2)/((1-w1)*N+1-p1))
    return(new1,new2)

def munew (x,tab,N):           # Attention ici, a choisir les bons sigma, (mettre a jour le tableau avt ou après est a décier ) , au pire je pourrais revoir le code pour un truc spécial            
    w1, mu1, sig1, mu2, sig2 = tab
    p1 = Prob(x,tab)
    new1 = (w1*N*mu1 + p1*x)/(w1*N +p1)
    new2 = ((1-w1)*N*mu2 + (1-p1)*x)/((1-w1)*N +(1-p1))
    return(new1,new2)
    



