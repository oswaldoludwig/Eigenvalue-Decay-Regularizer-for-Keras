#author: Oswaldo Ludwig

#This code implements a light version of the Eigenvalue Decay regularizer for 
#the Keras deep learning library, approximating the dominant eigenvalue by a 
#soft function given by the power method.

#The syntax for Eigenvalue Decay is similar to the other Keras weight regularizers, e.g.:

#from EigenvalueDecay import Regularizer
#from EigenvalueDecay import EigenvalueRegularizer
#...
#model.add(Dense(100,init='he_normal',W_regularizer=EigenvalueRegularizer(0.0005)))

#In case of publication using this code, please cite:

#Oswaldo Ludwig, Urbano Nunes, and Rui Araujo. "Eigenvalue decay: A new method for neural network regularization." Neurocomputing 124 (2014): 33-42.  

#and the version for deep learning:

#Oswaldo Ludwig. "Deep learning with Eigenvalue Decay regularizer." ArXiv preprint, 2016.


import theano.tensor as T
import numpy as np


class Regularizer(object):
    def set_param(self, p):
        self.p = p

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__}


class EigenvalueRegularizer(Regularizer):
    def __init__(self, k):
        self.k = k

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        W = self.p
        WW = T.dot(W.T,W)
        dim1, dim2 = WW.shape.eval() #The number of neurons in the layer
        k = self.k
        o = np.ones(dim1) #initial values for the dominant eigenvector
        
        #POWER METHOD FOR APPROXIMATING THE DOMINANT EIGENVECTOR (9 ITERATIONS): 
        domineigvec = T.dot(WW,T.dot(WW,T.dot(WW,T.dot(WW,T.dot(WW,T.dot(WW,T.dot(WW,T.dot(WW,T.dot(WW,o)))))))))
        
        WWd = T.dot(WW,domineigvec)
        domineigval = T.dot(WWd,domineigvec)/T.dot(domineigvec,domineigvec) #THE CORRESPONDING DOMINANT EIGENVALUE
        loss += (domineigval ** 0.5) * k #multiplied by the given regularization gain 
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "k": self.k}