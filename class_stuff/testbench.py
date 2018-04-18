import sys
sys.path.insert(0, './class_stuff')
import LMNN_SS
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split


dataset = load_iris()
X, y = dataset.data, dataset.target

# 105-45 samples in train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 30-75 labeled-unlabeled
X_labeled = np.delete(X_train, range(75), 0)
y_labeled = np.delete(y_train, range(75))
X_unlabeled = np.delete(X_train, range(75,105), 0)

# reload module in case there are changes to LMNN_SS.py
import importlib
importlib.reload(LMNN_SS)

model = LMNN_SS.SemiSupervisedLargeMarginNearestNeighbor().fit(X_labeled, X_unlabeled, y_labeled)


y_preds = model.predict(X_test)

model.score(y_test, y_preds)





# V's loss_SS
# 6.5622460569292489
# array([[ 0.03340182,  0.03733607, -0.07176629, -0.15372693],
#        [ 0.05829297,  0.05547538, -0.12174217, -0.25914227],
#        [-0.18014884, -0.1807451 ,  0.3830013 ,  0.80712692],
#        [-0.24553373, -0.25077481,  0.52783259,  1.11719199]])

# O's test-lmnn-impl
# 6.56209810987359
# array([[ 0.04118917,  0.04129688, -0.08899836, -0.18772006],
#        [ 0.05966518,  0.0605011 , -0.12575836, -0.2673461 ],
#        [-0.1730255 , -0.17579962,  0.37077371,  0.78278925],
#        [-0.24993164, -0.25359375,  0.53329826,  1.12832529]])

# K's LMNN_SS
# 4.614395969952108
# array([[ 0.03496919,  0.02878786, -0.07031082, -0.15409393],
#        [ 0.0643427 ,  0.03811899, -0.11557432, -0.25159979],
#        [-0.20435756, -0.13973847,  0.38943703,  0.84038951],
#        [-0.2818128 , -0.19480408,  0.53853121,  1.18075897]])