import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

# k-value here
k = 4

# data
dataset = load_iris()
X, y = dataset.data, dataset.target

# subset into labeled and unlabeled parts
X_labeled = np.delete(X,(range(40),range(50,90),range(100,140)),0)
y = np.delete(y,(range(40),range(50,90),range(100,140)))
X_unlabeled = np.delete(X,(range(40,50),range(90,100),range(140,150)),0)

# number of observations in labeled set
m = X_labeled.shape[0]
# number of observations in unlabeled set
m_ul = X_unlabeled.shape[0]

# number of dimensions
d = X_labeled.shape[1]

# initial distances - labeled to labeled - m by m
initial_distance_ll = euclidean_distances(X_labeled, squared=True)

# initial distances - labeled to unlabeled - m by m_un
initial_distance_ul = euclidean_distances(X_labeled,X_unlabeled, squared=True)

# find indexes of k closest same class points in initial space
# now saved as sparse matrix as well
dif_class_matrix = np.zeros(shape = (m,m))
eta_index = np.empty(shape = (m,k), dtype=int)
eta_row = np.repeat(range(m),k)
eta_col = []
for i in range(m):
    # take points that belong to different class than point i
    dif_class = y != y[i]
    dif_class_matrix[i,dif_class] = 1
    # take distances from point i to all other points
    point_distances = initial_distance_ll[i,]
    # assign large values to different class points - so they will never be closest ones
    point_distances[dif_class] = max(point_distances) * 10
    point_distances[i] = max(point_distances) * 10
    # remember indexes of k closest same class neighbours for point i
    eta_col = np.append(eta_col,point_distances.argsort()[0:k])
    eta_index[i,] = point_distances.argsort()[0:k]
eta = csr_matrix((np.repeat(1,m*k), (eta_row, eta_col)), shape=(m, m))

# same procedure to find closest unlabeled points
eta_index_ul = np.empty(shape = (m,k), dtype=int)
eta_row_ul = np.repeat(range(m),k)
eta_col_ul = []
for i in range(m):
    # take distances from point i to all other unlabeled points
    point_distances = initial_distance_ul[i,]
    # remember indexes of k closest unlabeled neighbours for point i
    eta_col_ul = np.append(eta_col_ul,point_distances.argsort()[0:k])
    eta_index_ul[i,] = point_distances.argsort()[0:k]
eta_ul = csr_matrix((np.repeat(1,m*k), (eta_row_ul, eta_col_ul)), shape=(m, m_ul))

# define omegas here:
omega_1 = 0.5
omega_2 = 0.5
omega_3 = 0.5


def loss_ss(l_in):
    # transform L from vector back to matrix
    _L = np.reshape(l_in,(d,d))

    # calculate linear transformations X L'
    _X_transformed = np.dot(np.asmatrix(X_labeled), np.asmatrix(_L).T)
    _X_transformed_ul = np.dot(np.asmatrix(X_unlabeled), np.asmatrix(_L).T)

    # calculate pairwise distances between all samples in transformed X
    _distance_matrix = euclidean_distances(_X_transformed, squared=True)
    _distance_matrix_ul = euclidean_distances(_X_transformed,_X_transformed_ul, squared=True)

    # LMNN loss function

    # PULL step - labeled part
    _neighbour_distance_matrix = eta.multiply(_distance_matrix)
    _pull_sum = omega_1 * _neighbour_distance_matrix.sum()

    # Pull step - unlabeled part
    _neighbour_distance_matrix_ul = eta_ul.multiply(_distance_matrix_ul)
    _pull_sum_ul = omega_2 * _neighbour_distance_matrix_ul.sum()

    # auxiliary distances to DIFFERENT CLASS points in labeled dataset
    _distance_matrix_aux = np.multiply(_distance_matrix,dif_class_matrix)
    _distance_matrix_aux[_distance_matrix_aux == 0] = 100 * np.amax(_distance_matrix_aux)

    # PUSH step - labeled part
    _impostor_num = 0
    _i_indexes = []
    _j_indexes = []
    _impostor_indexes = []
    for i in range(m):
        for j in eta_index[i,]:
            _reference_distance = _distance_matrix[i,j] + 1 # distance_matrix[i,j] + 1
            _impostor_num += sum(_distance_matrix_aux[i,] <= _reference_distance)
            _impostors = [j for j, x in enumerate(_distance_matrix_aux[i,] <= _reference_distance) if x]
            _i_indexes = np.append(_i_indexes,np.repeat(i,np.size(_impostors)))
            _j_indexes = np.append(_j_indexes, np.repeat(j, np.size(_impostors)))
            _impostor_indexes = np.append(_impostor_indexes,_impostors)

    _i_indexes = _i_indexes.astype(int)
    _j_indexes = _j_indexes.astype(int)
    _impostor_indexes = _impostor_indexes.astype(int)

    _push_sum = 0
    for n in range(_impostor_num):
        _i = _i_indexes[n]
        _j = _j_indexes[n]
        _l = _impostor_indexes[n]
        _push_sum += (1 + _distance_matrix[_i,_j] - _distance_matrix[_i,_l])

    _push_sum = (1 - omega_1) * _push_sum

    # PUSH step - unlabeled part
    _impostor_num_ul = 0
    _i_indexes_ul = []
    _j_indexes_ul = []
    _impostor_indexes_ul = []
    for i in range(m):
        for j in eta_index_ul[i,]:
            _reference_distance = _distance_matrix_ul[i, j] + 1
            _impostor_num_ul += sum(_distance_matrix_aux[i,] <= _reference_distance)
            _impostors_ul = [j for j, x in enumerate(_distance_matrix_aux[i,] <= _reference_distance) if x]
            _i_indexes_ul = np.append(_i_indexes_ul, np.repeat(i, np.size(_impostors_ul)))
            _j_indexes_ul = np.append(_j_indexes_ul, np.repeat(j, np.size(_impostors_ul)))
            _impostor_indexes_ul = np.append(_impostor_indexes_ul, _impostors_ul)

    _i_indexes_ul = _i_indexes_ul.astype(int)
    _j_indexes_ul = _j_indexes_ul.astype(int)
    _impostor_indexes_ul = _impostor_indexes_ul.astype(int)

    _push_sum_ul = 0
    for n in range(_impostor_num_ul):
        _i_ul = _i_indexes_ul[n]
        _j_ul = _j_indexes_ul[n]
        _l_ul = _impostor_indexes_ul[n]
        _push_sum_ul += (1 + _distance_matrix_ul[_i_ul,_j_ul] - _distance_matrix[_i_ul,_l_ul])

    _push_sum_ul = (1 - omega_2) * _push_sum_ul

    _total_loss = omega_3 *(_pull_sum + _push_sum) + (1 - omega_3) * (_pull_sum_ul + _push_sum_ul)

    print(_total_loss)

    return _total_loss


def loss_ss_jac(l_in):

    # transform L from vector back to matrix
    _L = np.reshape(l_in,(d,d))

    # calculate linear transformation X L'
    _X_transformed = np.dot(np.asmatrix(X_labeled), np.asmatrix(_L).T)
    _X_transformed_ul = np.dot(np.asmatrix(X_unlabeled), np.asmatrix(_L).T)

    # calculate pairwise distances between all samples in transformed X
    _distance_matrix = euclidean_distances(_X_transformed, squared=True)
    _distance_matrix_ul = euclidean_distances(_X_transformed,_X_transformed_ul, squared=True)

    # auxiliary distances to DIFFERENT CLASS points
    _distance_matrix_aux = np.multiply(_distance_matrix,dif_class_matrix)
    _distance_matrix_aux[_distance_matrix_aux == 0] = 10 * np.amax(_distance_matrix_aux)

    # calculate jacobian elementwise
    _jac = np.zeros((d,d))
    _jac_ul = np.zeros((d, d))

    # labeled part
    # most stupid loops ever
    for i in range(m):
        for j in range(k):
            # index of k-th nearest same class neighbour of i
            _index_j = eta_index[i,j]
            # reference distance to deciding if there is a push-step derivative at all
            _reference_distance = _distance_matrix[i,_index_j] + 1
            # find possible impostors
            _impostors = [z for z, x in enumerate(_distance_matrix_aux[i,] < _reference_distance) if x]
            # if there are impostors for given pair x_i and x_j
            if np.size(_impostors) > 0:
                for imp in _impostors:
                    _p1 = 2 * np.dot((_X_transformed[i,].T - _X_transformed[_index_j,].T),np.reshape((X_labeled[i,] - X_labeled[_index_j,]),(1,d)))
                    _p2 = 2 * np.dot((_X_transformed[i,].T - _X_transformed[imp,].T),np.reshape((X_labeled[i,] - X_labeled[imp,]),(1,d)))
                    _jac += (1 - omega_1) * (_p1 - _p2)
                _jac += omega_1 * 2 * np.dot((_X_transformed[i,].T - _X_transformed[_index_j,].T),np.reshape((X_labeled[i,] - X_labeled[_index_j,]),(1,d)))
            # if there is NO impostors for given pair x_i and x_j
            else:
                _jac += omega_1 * 2 * np.dot((_X_transformed[i,].T - _X_transformed[_index_j,].T),np.reshape((X_labeled[i,] - X_labeled[_index_j,]),(1,d)))

    # unlabeled part
    for i in range(m):
        for j in range(k):
            # index of k-th nearest same class neighbour of i
            _index_j_ul = eta_index_ul[i,j]
            # reference distance to deciding if there is a push-step derivative at all
            _reference_distance_ul = _distance_matrix_ul[i,_index_j_ul] + 1
            # find possible impostors
            _impostors_ul = [z for z, x in enumerate(_distance_matrix_aux[i,] < _reference_distance_ul) if x]
            # if there are impostors for given pair x_i and x_j
            if np.size(_impostors_ul) > 0:
                for imp in _impostors_ul:
                    _p1 = 2 * np.dot((_X_transformed[i,].T - _X_transformed_ul[_index_j_ul,].T),np.reshape((X_labeled[i,] - X_unlabeled[_index_j_ul,]),(1,d)))
                    _p2 = 2 * np.dot((_X_transformed[i,].T - _X_transformed[imp,].T),np.reshape((X_labeled[i,] - X_labeled[imp,]),(1,d)))
                    _jac_ul += (1 - omega_2) * (_p1 - _p2)
                _jac_ul += omega_2 * 2 * np.dot((_X_transformed[i,].T - _X_transformed_ul[_index_j_ul,].T),np.reshape((X_labeled[i,] - X_unlabeled[_index_j_ul,]),(1,d)))
            # if there is NO impostors for given pair x_i and x_j
            else:
                _jac_ul += omega_2 * 2 * np.dot((_X_transformed[i,].T - _X_transformed_ul[_index_j_ul,].T),np.reshape((X_labeled[i,] - X_unlabeled[_index_j_ul,]),(1,d)))

    _jac_total = omega_3 * _jac + (1 - omega_3) * _jac_ul
    #_jac_total = np.reshape(_jac_total,(1,d**2))

    return _jac_total

def loss_grad(L_in):
    loss = loss_ss(L_in)
    grad = loss_ss_jac(L_in)

    return loss, grad.flatten()



L_init = np.eye(d)
import scipy.optimize

L, loss, details = scipy.optimize.fmin_l_bfgs_b(func=loss_grad, x0=L_init.flatten(),
                                                maxfun=50, maxiter=20, pgtol=0.001)

