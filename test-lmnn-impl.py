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

# subset into labeled and unlabeled parts
X_labeled = np.delete(X,(range(40),range(50,90),range(100,140)),0)
labels = np.delete(labels,(range(40),range(50,90),range(100,140)))
X_unlabeled = np.delete(X,(range(40,50),range(90,100),range(140,150)),0)


m = X_labeled.shape[0]
m_ul = X_unlabeled.shape[0]
d = X.shape[1]
k = 4
omega_1 = 0.5
omega_2 = 0.5
omega_3 = 0.5

initial_distance_l = euclidean_distances(X_labeled, squared=True)

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

eta = compute_eta(initial_distance_l).todense()

def compute_eta_ul(initial_distance_ul):
    # same procedure to find closest unlabeled points
    eta_index_ul = np.empty(shape=(m, k), dtype=int)
    eta_row_ul = np.repeat(range(m), k)
    eta_col_ul = []
    for i in range(m):
        # take distances from point i to all other unlabeled points
        point_distances = initial_distance_ul[i,]
        # remember indexes of k closest unlabeled neighbours for point i
        eta_col_ul = np.append(eta_col_ul, point_distances.argsort()[0:k])
        eta_index_ul[i,] = point_distances.argsort()[0:k]
    eta_ul = csr_matrix((np.repeat(1, m * k), (eta_row_ul, eta_col_ul)), shape=(m, m_ul))
    return eta_ul

initial_distance_ul = euclidean_distances(X_labeled, X_unlabeled, squared=True)

eta_ul = compute_eta_ul(initial_distance_ul).todense()

y = np.matrix([labels == labels[i] for i in range(m)], dtype="int")
np.fill_diagonal(y, 0)

#y = np.matrix([[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]])

def _compute_pull_lmnn(omega_1, eta, X_labeled_tr):

    sum = 0
    count = 0
    for i in range(m):
        for j in range(m):
            if eta[i, j] == 0:
                continue
            if j != i:
                dist_ij = norm(X_labeled_tr[i] - X_labeled_tr[j])**2
                count += 1
                sum += eta[i, j] * dist_ij
    #print("count pull loss: {}".format(count))
    return omega_1 * sum


def _compute_push_lmnn(omega_1, eta, L, X_labeled_tr, y):
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
                        dist_ij = norm(X_labeled_tr[i] - X_labeled_tr[j])**2
                        dist_il = norm(X_labeled_tr[i] - X_labeled_tr[l])**2
                        #dist_pos = max(0, 1 + dist_ij - dist_il)
                        #addend = eta[i, j] * (1 - y[i, l]) * dist_pos

                        if 1 + dist_ij - dist_il > 0:
                            addend = eta[i,j] * (1 - y[i,l]) * (1 + dist_ij - dist_il)
                            total += addend
                            count += 1
                            impostor_indices.append((i, j, l))
                            # print("Adding {} to sum in loss (i,j,l) = ({}, {}, {}) ".format(addend, i, j, l))

    #print("count loss : {}".format(count))
    return (1 - omega_1) * total, impostor_indices

# old args: w_1, eta, L, X, y
def compute_loss_lmnn(L, X_labeled_tr, eta):
    pull = _compute_pull_lmnn(omega_1, eta, X_labeled_tr)
    push, impostor_indices = _compute_push_lmnn(omega_1, eta, L, X_labeled_tr, y)

    #print("loss pull L {}".format(pull))
    #print("loss push L {}".format(push))

    #print("Loss: {}".format(pull + push))
    return pull + push, impostor_indices

def norm_grad_g_ij_lmnn(i, j, L, X_labeled, X_labeled_tr):
    g = np.empty([d, d])
    for n in range(d):
        for k in range(d):

            grad_add = 2 * (X_labeled_tr[i,n] - X_labeled_tr[j,n]) * (X_labeled[i, k] - X_labeled[j, k])
            #print("Adding {} (n,k) = ({}, {})".format(grad_add, n, k))

            g[n,k] = grad_add
    return g

def norm_grad_g_il_lmnn(i, l, L, X_labeled, X_labeled_tr):
    g = np.empty([d, d])
    for n in range(d):
        for k in range(d):

            grad_add = 2 * (X_labeled_tr[i,n] - X_labeled_tr[l,n]) * (X_labeled[i, k] - X_labeled[l, k])
            #print("Adding {} (n,k) = ({}, {})".format(grad_add, n, k))

            g[n,k] = grad_add
    return g


def _compute_gradient_pull_lmnn(omega_1, eta, L, X_labeled, X_labeled_tr):
    total = np.zeros([d,d])
    count = 0
    for i in range(m):
        for j in range(m):
            if eta[i, j] == 0:
                continue
            if j != i:
                count += 1
                total += eta[i,j] * norm_grad_g_ij_lmnn(i, j, L, X_labeled, X_labeled_tr)

    #print("count pull grad: {}".format(count))
    return omega_1 * total

def _compute_gradient_push_lmnn(omega_1, eta, L, X_labeled, X_labeled_tr, y):
    total = np.zeros([d, d])

    count = 0
    for i in range(m):
        for j in range(m):
            if j != i:
                for l in range(m):
                    if eta[i,j] == 0 or (1 - y[i,l]) == 0:
                        continue
                    if l !=i and l != j:
                        dist_ij = norm(X_labeled_tr[i] - X_labeled_tr[j])** 2
                        dist_il = norm(X_labeled_tr[i] - X_labeled_tr[l])** 2

                        if 1 + dist_ij - dist_il > 0:

                            g_ij = norm_grad_g_ij_lmnn(i, j, L, X_labeled, X_labeled_tr)
                            g_il = norm_grad_g_il_lmnn(i, l, L, X_labeled, X_labeled_tr)
                            g_sum = g_ij - g_il
                            addend = eta[i,j] * (1 - y[i,l]) * g_sum

                            #if sum(sum(addend)) > 60:
                            #    print("Adding {} to sum (i,j,l) = ({},{},{})".format(sum(sum(addend)), i, j, l))
                            count += 1
                            total += addend

    #print("count gradient : {}".format(count))

    return (1 - omega_1) * total

# for every point pair(pull) and triplet (push), there's a d^2-element gradient
# old args: w_1, eta, L, X, y
def compute_gradient_lmnn(L, X_labeled, X_labeled_tr, eta):

    pull = _compute_gradient_pull_lmnn(omega_1, eta, L, X_labeled, X_labeled_tr)
    push = _compute_gradient_push_lmnn(omega_1, eta, L, X_labeled, X_labeled_tr, y)

    #print("pull:\n{},\npush:\n{}".format(pull, push))

    #print("GRAD:\n{},\n".format(np.reshape(pull + push, (4, 4))))

    return pull + push

np.set_printoptions(suppress=True)

def loss_and_grad_lmnn(L):
    #global eta
    L = np.reshape(L, [d,d])
    X_tr = np.matmul(X_labeled, L.T)

    #distances = euclidean_distances(X_tr, squared=True)
    #new_eta = compute_eta(distances).todense()
    #eta_diff = eta - new_eta
    #print("ETA DIFF: {}".format(sum(sum(eta_diff).transpose())))
    #eta = new_eta

    loss = compute_loss_lmnn(L, X_tr, eta)
    grad = compute_gradient_lmnn(L, X_tr, eta)

    return loss, grad.flatten()


def _compute_pull_ul(omega_2, eta_ul, X_labeled_tr, X_unlabeled_tr):
    sum = 0
    count = 0
    for i in range(m):
        for j in range(m_ul):
            if eta_ul[i, j] == 0:
                continue
            dist_ij = norm(X_labeled_tr[i] - X_unlabeled_tr[j])**2
            count += 1
            sum += eta_ul[i, j] * dist_ij
    #print("count pull loss UL: {}".format(count))
    return omega_2 * sum

def _compute_push_ul(omega_2, eta_ul, X_labeled_tr, X_unlabeled_tr):
    total = 0
    impostor_indices = []

    count = 0

    for i in range(m):
        for j in range(m_ul):
            for l in range(m):
                if eta_ul[i, j] == 0 or (1 - y[i, l]) == 0:
                    continue
                if l != i :
                    dist_ij = norm(X_labeled_tr[i] - X_unlabeled_tr[j]) ** 2
                    dist_il = norm(X_labeled_tr[i] - X_labeled_tr[l]) ** 2
                    # dist_pos = max(0, 1 + dist_ij - dist_il)
                    # addend = eta[i, j] * (1 - y[i, l]) * dist_pos

                    if 1 + dist_ij - dist_il > 0:
                        addend = eta_ul[i, j] * (1 - y[i, l]) * (1 + dist_ij - dist_il)
                        total += addend
                        count += 1
                        #impostor_indices.append((i, j, l))
                        # print("Adding {} to sum in loss (i,j,l) = ({}, {}, {}) ".format(addend, i, j, l))

    #print("count push UL loss : {}".format(count))
    return (1 - omega_2) * total


def compute_loss_ul(L, X_labeled_tr, X_unlabeled_tr, eta_ul):
    pull = _compute_pull_ul(omega_2, eta_ul, X_labeled_tr, X_unlabeled_tr)
    push = _compute_push_ul(omega_2, eta_ul, X_labeled_tr, X_unlabeled_tr)

    #print("loss pull UL {}".format(pull))
    #print("loss push UL {}".format(push))

    return pull + push


def compute_loss_ssc(L, X_labeled_tr, X_unlabeled_tr, eta, eta_ul):
    loss_lmnn, _ = compute_loss_lmnn(L, X_labeled_tr, eta)

    loss_ul = compute_loss_ul(L, X_labeled_tr, X_unlabeled_tr, eta_ul)

    total = (1 - omega_3) * loss_lmnn + omega_3 * loss_ul

    #print("loss ssc: {}".format(total))
    return total

def norm_grad_g_ij_ssc(i, j, X_labeled, X_unlabeled, X_labeled_tr, X_unlabeled_tr):
    g = np.empty([d, d])
    for n in range(d):
        for k in range(d):

            grad_add = 2 * (X_labeled_tr[i,n] - X_unlabeled_tr[j,n]) * (X_labeled[i, k] - X_unlabeled[j, k])
            #print("Adding {} (n,k) = ({}, {})".format(grad_add, n, k))

            g[n,k] = grad_add
    return g

def norm_grad_g_il_ssc(i, l, X_labeled, X_unlabeled, X_labeled_tr, X_unlabeled_tr):
    g = np.empty([d, d])
    for n in range(d):
        for k in range(d):

            grad_add = 2 * (X_labeled_tr[i,n] - X_labeled_tr[l,n]) * (X_labeled[i, k] - X_labeled[l, k])
            #print("Adding {} (n,k) = ({}, {})".format(grad_add, n, k))

            g[n,k] = grad_add
    return g

def _compute_gradient_pull_ul(eta_ul, X_labeled, X_unlabeled, X_labeled_tr, X_unlabeled_tr):
    total = np.zeros([d, d])
    count = 0
    for i in range(m):
        for j in range(m_ul):
            if eta_ul[i, j] == 0:
                continue
            count += 1
            total += eta_ul[i, j] * norm_grad_g_ij_ssc(i, j, X_labeled, X_unlabeled, X_labeled_tr, X_unlabeled_tr)

    return omega_2 * total

def _compute_gradient_push_ul(eta_ul, X_labeled, X_unlabeled, X_labeled_tr, X_unlabeled_tr):
    total = np.zeros([d, d])

    count = 0
    for i in range(m):
        for j in range(m_ul):
            for l in range(m):
                if eta_ul[i, j] == 0 or (1 - y[i, l]) == 0:
                    continue
                if l != i:
                    dist_ij = norm(X_labeled_tr[i] - X_unlabeled_tr[j]) ** 2
                    dist_il = norm(X_labeled_tr[i] - X_labeled_tr[l]) ** 2

                    if 1 + dist_ij - dist_il > 0:
                        g_ij = norm_grad_g_ij_ssc(i, j, X_labeled, X_unlabeled, X_labeled_tr, X_unlabeled_tr)
                        g_il = norm_grad_g_il_ssc(i, l, X_labeled, X_unlabeled, X_labeled_tr, X_unlabeled_tr)
                        g_sum = g_ij - g_il
                        addend = eta_ul[i, j] * (1 - y[i, l]) * g_sum

                        count += 1
                        total += addend

    return (1 - omega_2) * total


def compute_gradient_ssc(L, X_labeled, X_unlabeled, X_labeled_tr, X_unlabeled_tr, eta, eta_ul):
    grad_lmnn = (1-omega_3) * compute_gradient_lmnn(L, X_labeled, X_labeled_tr, eta)

    pull = _compute_gradient_pull_ul(eta_ul, X_labeled, X_unlabeled, X_labeled_tr, X_unlabeled_tr)
    push = _compute_gradient_push_ul(eta_ul, X_labeled, X_unlabeled, X_labeled_tr, X_unlabeled_tr)

    print("grad L:\n{}".format(grad_lmnn))
    print("grad UL:\n{}".format(omega_3 * (pull + push)))

    grad = grad_lmnn + omega_3 * (pull + push)

    #print("grad SSC:\n{}".format(grad))

    return grad

def loss_and_grad_ssc(L):
    L = np.reshape(L, [d, d])

    X_labeled_tr = np.matmul(X_labeled, L.T)
    X_unlabeled_tr = np.matmul(X_unlabeled, L.T)

    loss = compute_loss_ssc(L, X_labeled_tr, X_unlabeled_tr, eta, eta_ul)

    print("LOSS: {}".format(loss))

    grad = compute_gradient_ssc(L, X_labeled, X_unlabeled, X_labeled_tr, X_unlabeled_tr, eta, eta_ul)

    #print("GRAD:\n{}".format(grad))

    return loss, grad.flatten()

#compute_loss_ssc(L_init, X_labeled_tr, X_unlabeled_tr, eta, eta_ul)

#compute_gradient_ssc(L_init, X_labeled, X_unlabeled, X_labeled_tr, X_unlabeled_tr, eta, eta_ul)

#compute_gradient_lmnn(L_init, X_labeled_tr, eta)

import scipy.optimize

L, loss, details = scipy.optimize.fmin_l_bfgs_b(func=loss_and_grad_ssc, x0=L_init.flatten(),
                                                maxfun=50, maxiter=20, pgtol=0.001)


# Valentin's:
# 6.5620757775746661
# array([[ 0.03448816,  0.03459457, -0.07281244, -0.15418275],
#        [ 0.05694719,  0.05773282, -0.12273826, -0.25848097],
#        [-0.17870136, -0.18231812,  0.38294122,  0.80592533],
#        [-0.2458167 , -0.25077252,  0.52878739,  1.11560778]])


# Oskar's:
# 6.56209810987359
# array([[ 0.04118917,  0.04129688, -0.08899836, -0.18772006],
#        [ 0.05966518,  0.0605011 , -0.12575836, -0.2673461 ],
#        [-0.1730255 , -0.17579962,  0.37077371,  0.78278925],
#        [-0.24993164, -0.25359375,  0.53329826,  1.12832529]])