import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances

# k-value here
k = 4

# data
dataset = load_iris()
X, y = dataset.data, dataset.target

# number of observations
m = X.shape[0]

# number of dimensions
d = X.shape[1]

# initial L - just identity
L_init = np.eye(d)
L_init = np.reshape(L_init, (d**2,))

# initial distances
initial_distance = euclidean_distances(X, squared=True)

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
    point_distances = initial_distance[i,]
    # assign large values to different class points - so they will never be closest ones
    point_distances[dif_class] = max(point_distances) * 10
    # remember indexes of k closest same class neighbours for point i
    eta_col = np.append(eta_col,point_distances.argsort()[1:k + 1])
    eta_index[i,] = point_distances.argsort()[1:k + 1]
eta = csr_matrix((np.repeat(1,m*k), (eta_row, eta_col)), shape=(m, m))

# NOW NOT NEEDED FUNCTION:
def impostors_distances(neighbour_k,distance_matrix):
    # neighbour_k - m by 1, consists of distances for all m points to their k-s nearest same class neighbour
    # distance _matrix - m by m

    distance_matrix_ = distance_matrix
    # create sparce matrix that would contain distances to impostors in regards to point i (row) and  ...
    # value provided in neighbour_k
    imp_row = []
    imp_col = []
    imp_data = []
    for i in range(m):
        # find same class points for point i
        same_class = y == y[i]
        # take dstances from point i to all other points
        point_distances = distance_matrix_[i,]
        # make sure that same class points are further away than k-s neighbour of point i - self included
        point_distances[same_class] = neighbour_k[i] + 1
        # remember indexes of points, that are closer to i than value in neighbour_k[i]
        impostors = [j for j, x in enumerate(point_distances < neighbour_k[i]) if x]
        imp_col = np.append(imp_col,impostors)
        imp_row = np.append(imp_row,np.repeat(i,len(impostors)))
        # remember value in neighbour_k[i] - impostor_distances
        imp_data = np.append(imp_data,(neighbour_k[i] - point_distances[impostors]))
    impostor_data = csr_matrix((imp_data,(imp_row,imp_col)),shape=(m,m))
    return impostor_data

def loss_simple(L_in):

    # transform L from vector back to matrix
    L = np.reshape(L_in,(d,d))

    # calculate linear transformation X L'
    X_transformed = np.dot(np.asmatrix(X), np.asmatrix(L).T)

    # calculate pairwise distances between all samples in transformed X
    distance_matrix = euclidean_distances(X_transformed, squared=True)

    # LMNN loss function
    omega_1 = 0.5

    # PULL - step
    neighbour_distance_matrix = eta.multiply(distance_matrix)
    pull_sum = omega_1 * neighbour_distance_matrix.sum()

    # auxillary distances to DIFFERENT CLASS points
    distance_matrix_aux = np.multiply(distance_matrix,dif_class_matrix)
    distance_matrix_aux[distance_matrix_aux == 0] = 10 * np.amax(distance_matrix_aux)

    impostor_num = 0
    i_indexes = []
    j_indexes = []
    impostor_indexes = []
    for i in range(m):
        for j in eta_index[i,]:
            reference_distance = distance_matrix[i,j] + 1 # distance_matrix[i,j] + 1
            impostor_num += sum(distance_matrix_aux[i,] <= reference_distance)
            impostors =  [j for j, x in enumerate(distance_matrix_aux[i,] <= reference_distance) if x]
            i_indexes = np.append(i_indexes,np.repeat(i,np.size(impostors)))
            j_indexes = np.append(j_indexes, np.repeat(j, np.size(impostors)))
            impostor_indexes = np.append(impostor_indexes,impostors)

    i_indexes = i_indexes.astype(int)
    j_indexes = j_indexes.astype(int)
    impostor_indexes = impostor_indexes.astype(int)

    push_sum = 0
    for n in range(impostor_num):
        i = i_indexes[n]
        j = j_indexes[n]
        l = impostor_indexes[n]
        push_sum += (1 + distance_matrix[i,j] - distance_matrix[i,l])

    # # Push step
    # # create m by k matrix, which will have distances to k closest neighbours for all m elements
    # reference_distances_array = neighbour_distance_matrix.toarray()
    # reference_distances_array.sort(axis=1)
    # reference_distances_array = reference_distances_array[:,range(m-k,m)]

    # push_sum = 0
    # # calculate push part in k steps
    # act = 0
    # for i in range(k):
    #     p = impostors_distances(reference_distances_array[:,i],distance_matrix).sum()
    #     if p > 0:
    #         act += 1
    #     push_sum += p

    push_sum = (1 - omega_1) * push_sum

    total_loss = push_sum + pull_sum

    return total_loss

# res = minimize(loss_simple, L_init, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
# L_optim = res.x.reshape((d,d))

def loss_simple_jac(L_in):

    # transform L from vector back to matrix
    L = np.reshape(L_in,(d,d))

    # calculate linear transformation X L'
    X_transformed = np.dot(np.asmatrix(X), np.asmatrix(L).T)

    # find pairwise distances in new space
    distance_matrix = euclidean_distances(X_transformed, squared=True)

    # auxillary distances to DIFFERENT CLASS points
    distance_matrix_aux = np.multiply(distance_matrix,dif_class_matrix)
    distance_matrix_aux[distance_matrix_aux == 0] = 10 * np.amax(distance_matrix_aux)

    # calculate jacobian elementwise
    jac = np.zeros((d,d))

    # omega weight here
    omega_1 = 0.5

    # most stupid loops ever
    for i in range(m):
        for j in range(k):
            # index of k-th nearest same class neighbour of i
            index_j = eta_index[i,j]
            # reference distance to deciding if there is a push-step derivative at all
            reference_distance = distance_matrix[i,index_j] + 1
            # find possible impostors
            impostors = [z for z, x in enumerate(distance_matrix_aux[i,] < reference_distance) if x]
            # if there are impostors for given pair x_i and x_j
            if np.size(impostors) > 0:
                for imp in impostors:
                    p1 = 2 * np.dot((X_transformed[i,].T - X_transformed[index_j,].T),np.reshape((X[i,] - X[index_j,]),(1,d)))
                    p2 = 2 * np.dot((X_transformed[i,].T - X_transformed[imp,].T),np.reshape((X[i,] - X[imp,]),(1,d)))
                    jac += (1 - omega_1) * (p1 - p2)
            # if there is NO impostors for given pair x_i and x_j
            else:
                jac += omega_1 * 2 * np.dot((X_transformed[i,].T - X_transformed[index_j,].T),np.reshape((X[i,] - X[index_j,]),(1,d)))

    jac = np.reshape(jac,(1,d**2))
    return jac

# initial L - just identity
L_init = np.eye(d)
L_init = np.reshape(L_init, (d**2,))

res = minimize(loss_simple, L_init, method='L-BFGS-B',jac=loss_simple_jac,options={'disp': True})
L_optim = res.x.reshape((d,d))

L_optim_flat = np.reshape(L_optim,(d**2,))
loss_simple(L_optim_flat)
loss_simple_jac(L_optim_flat)