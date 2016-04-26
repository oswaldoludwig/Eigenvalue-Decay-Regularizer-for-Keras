# Eigenvalue Decay Regularizer for Keras
Eigenvalue Decay is a new option of weight regularizer to the deep learning practitioner that aims at maximum-margin learning.
This code implements a light version of the Eigenvalue Decay regularizer for the Keras library. This version approximates the dominant eigenvalue by a soft function given by the power method. In case of publication using this package cite:

Oswaldo Ludwig, Urbano Nunes, and Rui Araujo. "Eigenvalue decay: A new method for neural network regularization." Neurocomputing 124 (2014): 33-42.  

and the paper on deep learning with Eigenvalue Decay: 

Oswaldo Ludwig. "Deep learning with Eigenvalue Decay regularizer." ArXiv eprint arXiv:1604.06985 [cs.LG], (2016).
https://arxiv.org/abs/1604.06985

INSTALLATION AND USE:

1) Download the file “EigenvalueDecay.py” and save it in the same folder as your Keras model;

2) Include the following lines in your model file to import Eigenvalue Decay:
 
	from EigenvalueDecay import Regularizer
	from EigenvalueDecay import EigenvalueRegularizer

3) The syntax for Eigenvalue Decay is similar to the other Keras weight regularizers, e.g.:

	 model.add(Dense(100, init='he_normal', W_regularizer=EigenvalueRegularizer(0.0005)))

This folder also provides three examples of models from the Keras repository including regularization with Eigenvalue Decay: “Experiment1.py”, “Experiment2.py”, “Experiment3.py”. To run in GPU you can call the code like this:

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high python Experiment1.py

These experiments are the same as reported in our last paper, cited above. The original models from Keras repository (without Eigenvalue Decay) can be downloaded here: https://github.com/fchollet/keras/tree/master/examples
