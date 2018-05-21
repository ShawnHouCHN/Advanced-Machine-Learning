from numpy.linalg import norm
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from sklearn.datasets import load_iris
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y, check_random_state
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import ParameterGrid,GridSearchCV,StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from LMNN import LargeMarginNearestNeighbor
from LMNN_SS import SemiSupervisedLargeMarginNearestNeighbor
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import pandas as pd
import time


def cross_validate_test(clf, X, y, cv=5):
    print ("###### Cross Validate Data Shape: ", X.shape)
    skf = StratifiedKFold(n_splits=cv)
    scores_list=[]
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train,y_train)
        scores_list.append(clf.score(X_test,y_test))
    print ("##### Average Performance: ", np.mean(scores_list))
    return scores_list

def get_USPS_training_data():
    data = load_svmlight_file('ml_data/usps/usps')
    return data[0], data[1]

def get_USPS_testing_data():
    data = load_svmlight_file('ml_data/usps/usps.t')
    return data[0], data[1]

X_train, y_train = get_USPS_training_data()
X_test, y_test = get_USPS_testing_data()
X_train=np.asarray(X_train.todense())
y_train=np.asarray(y_train)
X_test=np.asarray(X_test.todense())
y_test=np.asarray(y_test)
X_all=np.concatenate((X_train, X_test), axis=0)
y_all=np.concatenate((y_train, y_test), axis=0)

my_model = PCA(n_components=0.95, svd_solver='full')
X_all=my_model.fit_transform(X_all)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=2000, stratify=y_all, random_state=42)
X_train_labelled, X_train_rest, y_train_labelled, y_train_rest = train_test_split(X_train, y_train, train_size=200, stratify=y_train, random_state=42)
sample_inds=np.random.randint(X_train_rest.shape[0], size=4000)
X_train_unlabelled=X_train_rest[sample_inds, :]

print("X_train: {}".format(X_train.shape))
print("X_test: {}".format(X_test.shape))
print("X_train_labelled: {}".format(X_train_labelled.shape))
print("X_train_unlabelled: {}".format(X_train_unlabelled.shape))


##### LMNN TUNNING USPS ########  (Long execution!!)
#lmnn_param_list = [2,5,10,50,100]
lmnn_param_list = [2,5,10]
lmnn_param_grid = {'max_iter': lmnn_param_list}
lmnnclf = GridSearchCV(LargeMarginNearestNeighbor(), lmnn_param_grid, cv=5, n_jobs=20, verbose=3)
lmnnclf.fit(X_train_labelled, y_train_labelled)

print("LMNN Best parameters set found on development set:")
print(lmnnclf.best_params_)
print("Grid scores on development set:")
means = lmnnclf.cv_results_['mean_test_score']
stds = lmnnclf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, lmnnclf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

print("Detailed classification report:")
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
y_true, y_pred = y_test, lmnnclf.predict(X_test)
print(classification_report(y_true, y_pred))