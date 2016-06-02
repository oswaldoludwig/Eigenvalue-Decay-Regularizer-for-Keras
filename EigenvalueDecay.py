#author: Oswaldo Ludwig

#This code implements a light version of the Eigenvalue Decay regularizer for
#the Keras deep learning library, approximating the dominant eigenvalue by a
#soft function given by the power method. (It only works with Theano backend)

#The syntax for Eigenvalue Decay is similar to the other Keras weight regularizers, e.g.:

#from EigenvalueDecay import EigenvalueRegularizer
#...
#model.add(Dense(100,W_regularizer=EigenvalueRegularizer(0.0005)))

#In case of publication using this code, please cite:

#Oswaldo Ludwig, Urbano Nunes, and Rui Araujo. "Eigenvalue decay: A new method for neural network regularization." Neurocomputing 124 (2014): 33-42.

#and the version for deep learning:

#Oswaldo Ludwig. "Deep learning with Eigenvalue Decay regularizer." ArXiv preprint, 2016.


import numpy as np
from keras import backend as K
from keras.regularizers import Regularizer


class EigenvalueRegularizer(Regularizer):
    """This class implements the Eigenvalue Decay regularizer.
    
    Args:
        The constant that controls the regularization on the current layer
        ( see Section 3 of https://arxiv.org/abs/1604.06985 )

    Returns:
        The regularized loss (for the training data) and
        the original loss (for the validation data).
        
    """
    def __init__(self, k):
        self.k = k
        self.uses_learning_phase = True


    def set_param(self, p):
        self.p = p


    def __call__(self, loss):
        power = 9  # number of iterations of the power method
        W = self.p
        WW = K.dot(K.transpose(W), W)
        dim1, dim2 = K.eval(K.shape(WW))
        k = self.k
        o = np.ones(dim1)  # initial values for the dominant eigenvector

        # power method for approximating the dominant eigenvector:
        domin_eigenvect = K.dot(WW, o)
        for n in range(power - 1):
            domin_eigenvect = K.dot(WW, domin_eigenvect)    
        
        WWd = K.dot(WW, domin_eigenvect)
        domin_eigenval = K.dot(WWd, domin_eigenvect) / K.dot(domin_eigenvect, domin_eigenvect)  # the corresponding dominant eigenvalue
        regularized_loss = loss + (domin_eigenval ** 0.5) * self.k  # multiplied by the given regularization gain
        return K.in_train_phase(regularized_loss, loss)
    

    def get_config(self):
        return {"name": self.__class__.__name__,
                "k": self.k}
