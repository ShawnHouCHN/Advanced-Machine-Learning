from load_mnist import load_mnist
X_train, y_train = load_mnist('training')
X_test, y_test = load_mnist('testing')

from scipy.io import savemat
import numpy as np
import os
X = np.vstack((X_train, X_test))
y = np.vstack((y_train, y_test))
wd = os.path.dirname(__file__)
savemat(os.path.join(wd, 'MNIST.mat'), {'X' : X, 'y' : y})