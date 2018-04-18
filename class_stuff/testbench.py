from LMNN_SS import SemiSupervisedLargeMarginNearestNeighbor
from sklearn.datasets import load_iris
import numpy as np

# data
dataset = load_iris()
X, y = dataset.data, dataset.target

############ SS part ##
X_labeled = np.delete(X, (range(40), range(50, 90), range(100, 140)), 0)
y = np.delete(y, (range(40), range(50, 90), range(100, 140)))
X_unlabeled = np.delete(X,(range(40,50),range(90,100),range(140,150)),0)
#######################

nn = SemiSupervisedLargeMarginNearestNeighbor()

nn.fit(X, y)

c = nn.predict(X)
# print(c)

d = nn.score(y, c)
print(d)

e = nn.predict_proba(X)
print(e)
