# Eigenvalue Decay Regularizer for Keras
Eigenvalue Decay is a new option of weight regularizer to the deep learning practitioner that aims at maximum-margin learning.
This code implements a light version of the Eigenvalue Decay regularizer for the Keras library. This version approximates the dominant eigenvalue by a soft function given by the power method. In case of publication using this software package cite:

Oswaldo Ludwig, Urbano Nunes, and Rui Araujo. "Eigenvalue decay: A new method for neural network regularization." Neurocomputing 124 (2014): 33-42.  

and the paper on deep learning with Eigenvalue Decay: 

Oswaldo Ludwig. "Deep learning with Eigenvalue Decay regularizer." ArXiv eprint arXiv:1604.06985 [cs.LG], (2016).
https://www.researchgate.net/publication/301648136_Deep_Learning_with_Eigenvalue_Decay_Regularizer

INSTALLATION AND USE:

1) Download the file “EigenvalueDecay.py” and save it in the same folder as your Keras model;

2) Include the following line in your model file to import Eigenvalue Decay:
 
	from EigenvalueDecay import EigenvalueRegularizer

3) The syntax for Eigenvalue Decay is similar to the other Keras weight regularizers, e.g.:

	 model.add(Dense(100, W_regularizer=EigenvalueRegularizer(0.0005)))

This folder also provides an example of a model from the Keras repository including regularization with Eigenvalue Decay: “example.py”. To run in GPU you can call the code like this:

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high python example.py

After training, you have to save the trained weights, create/compile a similar model without Eingenvalue Decay and save this model. Then, you can use your trained weights with this model (see lines 78-110 of “example.py”).

The second example (example2.py) yields a larger gain in the accuracy by the use of Eigenvalue Decay: 2.71% of gain (averaged over 10 runs).

For comparison, the original models from Keras repository (without Eigenvalue Decay) can be downloaded here: https://github.com/fchollet/keras/tree/master/examples

(This code is only for Theano backend)
