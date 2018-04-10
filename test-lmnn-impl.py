from types import new_class

from numpy.linalg import norm
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.sparse import csr_matrix
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances



#X = np.matrix([[3, 1], [4, 2], [14, 10], [13, 9]])
#labels = np.array([0,0,1,1])
#colors = ['red', 'red', 'blue', 'blue']
## y_i,l = 1 if same class
## eta_i,j = 1 if j is neighbour of i and they belong to same class
#eta = np.matrix([[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]])

dataset = load_iris()
X, labels = dataset.data, dataset.target
m = X.shape[0]
d = X.shape[1]
k = 4
omega_1 = 0.5

initial_distance = euclidean_distances(X, squared=True)

L_init = np.eye(d, d)

def compute_eta(distances):
    dif_class_matrix = np.zeros(shape=(m, m))
    eta_index = np.empty(shape=(m, k), dtype=int)
    eta_row = np.repeat(range(m), k)
    eta_col = []
    for i in range(m):
        dif_class = labels != labels[i]
        dif_class_matrix[i, dif_class] = 1
        point_distances = distances[i,]
        point_distances[dif_class] = max(point_distances) * 10
        point_distances[i] = max(point_distances) * 10
        # remember indexes of k closest same class neighbours for point i
        eta_col = np.append(eta_col, point_distances.argsort()[0:k])
        eta_index[i,] = point_distances.argsort()[0:k]
    eta = csr_matrix((np.repeat(1,m*k), (eta_row, eta_col)), shape=(m, m))
    return eta

eta = compute_eta(initial_distance).todense()

y = np.matrix([labels == labels[i] for i in range(m)], dtype="int")
np.fill_diagonal(y, 0)

#y = np.matrix([[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]])

def _compute_pull(omega_1, eta, X_tr):
    m = X_tr.shape[0]
    sum = 0
    count = 0
    for i in range(m):
        for j in range(m):
            if eta[i, j] == 0:
                continue
            if j != i:
                dist_ij = norm(X_tr[i] - X_tr[j])**2
                count += 1
                sum += eta[i, j] * dist_ij
    print("count pull loss: {}".format(count))
    return omega_1 * sum


def _compute_push(omega_1, eta, L, X_tr, y):
    count = 0
    total = 0
    impostor_indices = []

    count = 0

    for i in range(m):
        for j in range(m):
            if j != i:
                for l in range(m):
                    if eta[i,j] == 0 or (1 - y[i,l]) == 0:
                        continue
                    if l != i and l != j:
                        dist_ij = norm(X_tr[i] - X_tr[j])**2
                        dist_il = norm(X_tr[i] - X_tr[l])**2
                        #dist_pos = max(0, 1 + dist_ij - dist_il)
                        #addend = eta[i, j] * (1 - y[i, l]) * dist_pos

                        if 1 + dist_ij - dist_il > 0:
                            addend = eta[i,j] * (1 - y[i,l]) * (1 + dist_ij - dist_il)
                            total += addend
                            count += 1
                            impostor_indices.append((i, j, l))
                            # print("Adding {} to sum in loss (i,j,l) = ({}, {}, {}) ".format(addend, i, j, l))

    print("count loss : {}".format(count))
    return (1 - omega_1) * total, impostor_indices

# old args: w_1, eta, L, X, y
def compute_loss(L, X_tr, eta):
    pull = _compute_pull(omega_1, eta, X_tr)
    push, impostor_indices = _compute_push(omega_1, eta, L, X_tr, y)

    print("push: {}, pull: {}".format(push, pull))
    #print("Loss: {}".format(pull + push))
    return pull + push, impostor_indices

def norm_grad_g_ij(i, j, L, X, X_tr):
    g = np.empty([d, d])
    for n in range(d):
        for k in range(d):

            grad_add = 2 * (X_tr[i,n] - X_tr[j,n]) * (X[i, k] - X[j, k])
            #print("Adding {} (n,k) = ({}, {})".format(grad_add, n, k))

            g[n,k] = grad_add
    return g

def norm_grad_g_il(i, l, L, X, X_tr):
    g = np.empty([d, d])
    for n in range(d):
        for k in range(d):

            grad_add = 2 * (X_tr[i,n] - X_tr[l,n]) * (X[i, k] - X[l, k])
            #print("Adding {} (n,k) = ({}, {})".format(grad_add, n, k))

            g[n,k] = grad_add
    return g


def _compute_gradient_pull(omega_1, eta, L, X, X_tr):
    total = np.zeros([d,d])
    count = 0
    for i in range(m):
        for j in range(m):
            if eta[i, j] == 0:
                continue
            if j != i:
                count += 1
                total += eta[i,j] * norm_grad_g_ij(i, j, L, X, X_tr)

    print("count pull grad: {}".format(count))
    return omega_1 * total

def _compute_gradient_push(omega_1, eta, L, X, X_tr, y):
    total = np.zeros([d, d])

    count = 0
    for i in range(m):
        for j in range(m):
            if j != i:
                for l in range(m):
                    if eta[i,j] == 0 or (1 - y[i,l]) == 0:
                        continue
                    if l !=i and l != j:
                        dist_ij = norm(X_tr[i] - X_tr[j])** 2
                        dist_il = norm(X_tr[i] - X_tr[l])** 2

                        if 1 + dist_ij - dist_il > 0:

                            g_ij = norm_grad_g_ij(i, j, L, X, X_tr)
                            g_il = norm_grad_g_il(i, l, L, X, X_tr)
                            g_sum = g_ij - g_il
                            addend = eta[i,j] * (1 - y[i,l]) * g_sum

                            #if sum(sum(addend)) > 60:
                            #    print("Adding {} to sum (i,j,l) = ({},{},{})".format(sum(sum(addend)), i, j, l))
                            count += 1
                            total += addend

    print("count gradient : {}".format(count))

    return (1 - omega_1) * total

# for every point pair(pull) and triplet (push), there's a d^2-element gradient
# old args: w_1, eta, L, X, y
def compute_gradient(L, X_tr, eta):

    pull = _compute_gradient_pull(omega_1, eta, L, X, X_tr)
    push = _compute_gradient_push(omega_1, eta, L, X, X_tr, y)

    #print("pull:\n{},\npush:\n{}".format(pull, push))

    print("GRAD:\n{},\n".format(np.reshape(pull + push, (4, 4))))

    return pull + push

def plot(X, colors):
    plt.scatter(np.asarray(X[:, 1]), np.asarray(X[:, 0]), c=colors)


np.set_printoptions(suppress=True)

#compute_loss(L)
#compute_gradient(L_init)


def loss_and_grad(L):
    global eta
    L = np.reshape(L, [d,d])
    X_tr = np.matmul(X, L.T)

    distances = euclidean_distances(X_tr, squared=True)
    new_eta = compute_eta(distances).todense()
    eta_diff = eta - new_eta

    print("ETA DIFF: {}".format(sum(sum(eta_diff).transpose())))
    eta = new_eta

    loss = compute_loss(L, X_tr, eta)
    grad = compute_gradient(L, X_tr, eta)

    return loss, grad.flatten()


import scipy.optimize

#X_tr = np.matmul(X, L_init.T)
#loss, impostor_indices = compute_loss(L_init, X_tr, eta)
#m = set(impostor_indices)


#v = set(v_indices)
#i_s = v.intersection(m)
#unique_m = m - i_s
#unique_v = v - i_s

L, loss, details = scipy.optimize.fmin_l_bfgs_b(func=loss_and_grad, maxfun=20, maxls=5,
                                                x0=L_init, maxiter=10, pgtol=0.01,
                                                iprint=-1)
L = np.reshape(L, [d,d])

# 3637
# 804.8899999999994, pull: 43.75500000000001


# v - m
# {(142, 142, 78), (142, 142, 55), (142, 142, 83), (142, 142, 77), (142, 142, 54),
#  (54, 76, 103), (106, 113, 93),
# (142, 142, 66),
# (128, 111, 70), (71, 97, 123), (127, 149, 94), (51, 86, 116),
#  (142, 142, 85), (142, 142, 84), (142, 142, 70), (142, 142, 56), (142, 142, 68), (142, 142, 63), (142, 142, 91),
#  (54, 76, 141), (66, 96, 146), (116, 147, 91), (51, 75, 137),
#  (142, 142, 73), (142, 142, 72)}

# loss 152.92237
# [ 0.07906213,  0.40198965, -0.45669375, -1.03440687],
#        [ 0.12463876,  0.44083368, -0.57865573, -1.20659245],
#        [-0.30278305, -1.18589183,  1.47158055,  3.2036171 ],
#        [-0.33346051, -1.23635023,  1.5694044 ,  3.27713416]]

import sys
sys.path.insert(0, './pylmnn-master')
import pylmnn
from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN
from pylmnn.plots import plot_comparison


#plot_comparison(L, X, labels, dim_pref=3)

L_v = np.asarray([[ 1.66506679,  0.23159616,  0.05130396, -0.31491682],
       [ 0.21348119,  1.3123109 , -0.4564415 ,  0.20844194],
       [-0.06307738, -0.46742118,  1.3057125 ,  0.7407108 ],
       [-0.34575282,  0.18681816,  0.61021362,  1.63416581]])

L_m1 = np.asarray([[ 0.07906213,  0.40198965, -0.45669375, -1.03440687],
       [ 0.12463876,  0.44083368, -0.57865573, -1.20659245],
       [-0.30278305, -1.18589183,  1.47158055,  3.2036171 ],
       [-0.33346051, -1.23635023,  1.5694044 ,  3.27713416]])
#plot_comparison(L_v, X, labels, dim_pref=3)

# push: 807, 266, 240, 203, 189, 156, 132, 142, 127, 131, 128, 126, 126, 125
# pull:  43,  80,  82,  86,  82,  51,  60,  42,  40,  27,  28,  27,  27,  27