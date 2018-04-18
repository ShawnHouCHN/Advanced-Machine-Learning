import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize, fmin_l_bfgs_b


class SemiSupervisedLargeMarginNearestNeighbor(KNeighborsClassifier):

    def __init__(self, L=None, n_neighbors=4, method='L-BFGS-B', options={'disp': True},
                 omega=np.array([0.5, 0.5, 0.5]), max_iter=200, tol=1e-5):

        super(SemiSupervisedLargeMarginNearestNeighbor, self).__init__(n_neighbors=n_neighbors)

        # Number of neighbors
        self.k = n_neighbors

        self.method = method
        self.options = options
        self.options['maxiter'] = max_iter
        self.omega = omega
        self.max_iter = max_iter

        self.L_init = L
        self.L = L

        self.tol = tol

        # Defining the "global" variables in the class, some will hopefully disappear during code optimization
        self.X_labeled = None
        self.X_unlabeled = None
        self.y = None
        self.m = None
        self.m_ul = None
        self.d = None
        self.dif_class_matrix = None
        self.eta_index = None
        self.eta_index_ul = None
        self.eta = None
        self.eta_ul = None

    def predict(self, X):

        # Check if fit had been called
        check_is_fitted(self, ['X_labeled', 'X_unlabeled', 'y'])
        # Apply the fitted metric L to the data X
        # X_transformed = X.dot(self.L.T)

        y_pred = super(SemiSupervisedLargeMarginNearestNeighbor, self).predict(self.transform(X))

        return y_pred

    def fit(self, X_labeled, X_unlabeled, y):

        self.X_labeled = X_labeled
        self.X_unlabeled = X_unlabeled
        self.y = y

        # number of observations in labeled set
        self.m = self.X_labeled.shape[0]
        # number of observations in unlabeled set
        self.m_ul = self.X_unlabeled.shape[0]

        # number of dimensions
        self.d = self.X_labeled.shape[1]

        # initial L - just identity
        if self.L_init is None:
            self.L_init = np.eye(self.d)

        self.L_init = np.reshape(self.L_init, (self.d**2,))

        # initial distances - labeled to labeled - m by m
        initial_distance_ll = euclidean_distances(self.X_labeled, squared=True)

        # initial distances - labeled to unlabeled - m by m_un
        initial_distance_ul = euclidean_distances(self.X_labeled, self.X_unlabeled, squared=True)

        # find indexes of k closest same class points in initial space
        # now saved as sparse matrix as well
        self.dif_class_matrix = np.zeros(shape=(self.m, self.m))
        self.eta_index = np.empty(shape=(self.m, self.k), dtype=int)
        eta_row = np.repeat(range(self.m), self.k)
        eta_col = []

        for i in range(self.m):
            # take points that belong to different class than point i
            dif_class = y != y[i]
            self.dif_class_matrix[i, dif_class] = 1
            # take distances from point i to all other points
            point_distances = initial_distance_ll[i,]
            # assign large values to different class points - so they will never be closest ones
            point_distances[dif_class] = max(point_distances) * 10
            # remember indexes of k closest same class neighbours for point i
            eta_col = np.append(eta_col, point_distances.argsort()[0:self.k])
            self.eta_index[i, ] = point_distances.argsort()[0:self.k]

        self.eta = csr_matrix((np.repeat(1, self.m*self.k), (eta_row, eta_col)), shape=(self.m, self.m))

        # same procedure to find closest unlabeled points
        self.eta_index_ul = np.empty(shape=(self.m, self.k), dtype=int)
        eta_row_ul = np.repeat(range(self.m), self.k)
        eta_col_ul = []

        for i in range(self.m):
            # take distances from point i to all other unlabeled points
            point_distances = initial_distance_ul[i, ]
            # remember indexes of k closest unlabeled neighbours for point i
            eta_col_ul = np.append(eta_col_ul, point_distances.argsort()[0:self.k])
            self.eta_index_ul[i, ] = point_distances.argsort()[0:self.k]

        self.eta_ul = csr_matrix((np.repeat(1, self.m * self.k), (eta_row_ul, eta_col_ul)), shape=(self.m, self.m_ul))

        #L = minimize(fun=self.loss_ss, x0=self.L_init, method=self.method, jac=self.loss_ss_jac,
        #             options=self.options)

       # L, loss, details = fmin_l_bfgs_b(func=self.loss_gradient, x0=self.L_init.flatten(), bounds=None,
       #                                  m=100, pgtol=self.tol, maxfun=500*self.max_iter,
        #                                 maxiter=self.max_iter, disp=4, callback=None)
        L, loss, details = fmin_l_bfgs_b(func=self.loss_gradient, x0=self.L_init.flatten(),
                                                        maxfun=50, maxiter=20, pgtol=0.001)

        L_optim = np.reshape(L, [self.d, self.d])

        self.L = L_optim
        self.loss = loss
        self.details = details

        super(SemiSupervisedLargeMarginNearestNeighbor, self).fit(self.transform(X_labeled), y)

        return self

    def loss_gradient(self, L_in):
        loss = self.loss_ss(L_in)
        grad = self.loss_ss_jac(L_in)

        return loss, grad.flatten()


    def transform(self, X):

        #if X is None:
        #    X = self.X
        #else:
        #    X = check_array(X)

        return np.dot(np.asmatrix(X), np.asmatrix(self.L).T)

    def loss_ss(self, L_in):

        # transform L from vector back to matrix
        self.L = np.reshape(L_in, (self.d, self.d))

        # calculate linear transformation X L'
        # X_transformed = np.dot(np.asmatrix(self.X_labeled), np.asmatrix(L).T)
        # X_transformed_ul = np.dot(np.asmatrix(self.X_unlabeled), np.asmatrix(L).T)

        # calculate pairwise distances between all samples in transformed X
        distance_matrix = euclidean_distances(self.transform(self.X_labeled), squared=True)
        distance_matrix_ul = euclidean_distances(self.transform(self.X_labeled), self.transform(self.X_unlabeled), squared=True)

        # LMNN loss function

        # PULL step - labeled part
        neighbour_distance_matrix = self.eta.multiply(distance_matrix)
        pull_sum = self.omega[0] * neighbour_distance_matrix.sum()

        # PULL step - unlabeled part
        neighbour_distance_matrix_ul = self.eta_ul.multiply(distance_matrix_ul)
        pull_sum_ul = self.omega[1] * neighbour_distance_matrix_ul.sum()

        # auxillary distances to DIFFERENT CLASS points in labeled dataset
        distance_matrix_aux = np.multiply(distance_matrix, self.dif_class_matrix)
        distance_matrix_aux[distance_matrix_aux == 0] = 100 * np.amax(distance_matrix_aux)

        # PUSH step - labeled part
        impostor_num = 0
        i_indexes = []
        j_indexes = []
        impostor_indexes = []
        for i in range(self.m):
            for j in self.eta_index[i, ]:
                reference_distance = distance_matrix[i, j] + 1  # distance_matrix[i,j] + 1
                impostor_num += sum(distance_matrix_aux[i, ] <= reference_distance)
                impostors = [j for j, x in enumerate(distance_matrix_aux[i, ] <= reference_distance) if x]
                i_indexes = np.append(i_indexes, np.repeat(i, np.size(impostors)))
                j_indexes = np.append(j_indexes, np.repeat(j, np.size(impostors)))
                impostor_indexes = np.append(impostor_indexes, impostors)

        i_indexes = i_indexes.astype(int)
        j_indexes = j_indexes.astype(int)
        impostor_indexes = impostor_indexes.astype(int)

        push_sum = 0
        for n in range(impostor_num):
            i = i_indexes[n]
            j = j_indexes[n]
            l = impostor_indexes[n]
            push_sum += (1 + distance_matrix[i, j] - distance_matrix[i, l])

        push_sum = (1 - self.omega[0]) * push_sum

        # PUSH step - unlabeled part
        impostor_num_ul = 0
        i_indexes_ul = []
        j_indexes_ul = []
        impostor_indexes_ul = []
        for i in range(self.m):
            for j in self.eta_index_ul[i, ]:
                reference_distance = distance_matrix_ul[i, j] + 1
                impostor_num_ul += sum(distance_matrix_aux[i, ] <= reference_distance)
                impostors_ul = [j for j, x in enumerate(distance_matrix_aux[i, ] <= reference_distance) if x]
                i_indexes_ul = np.append(i_indexes_ul, np.repeat(i, np.size(impostors_ul)))
                j_indexes_ul = np.append(j_indexes_ul, np.repeat(j, np.size(impostors_ul)))
                impostor_indexes_ul = np.append(impostor_indexes_ul, impostors_ul)

        i_indexes_ul = i_indexes_ul.astype(int)
        j_indexes_ul = j_indexes_ul.astype(int)
        impostor_indexes_ul = impostor_indexes_ul.astype(int)

        push_sum_ul = 0
        for n in range(impostor_num_ul):
            i_ul = i_indexes_ul[n]
            j_ul = j_indexes_ul[n]
            l_ul = impostor_indexes_ul[n]
            push_sum_ul += (1 + distance_matrix_ul[i_ul, j_ul] - distance_matrix[i_ul, l_ul])

        push_sum_ul = (1 - self.omega[1]) * push_sum_ul

        total_loss = self.omega[2] * (pull_sum + push_sum) + (1 - self.omega[2]) * (pull_sum_ul + push_sum_ul)

        return total_loss

    def loss_ss_jac(self, L_in):

        # transform L from vector back to matrix
        # L = np.reshape(L_in, (self.d, self.d))

        # calculate linear transformation X L'
        X_transformed = self.transform(self.X_labeled)
        X_transformed_ul = self.transform(self.X_unlabeled)

        # calculate pairwise distances between all samples in transformed X
        distance_matrix = euclidean_distances(X_transformed, squared=True)
        distance_matrix_ul = euclidean_distances(X_transformed, X_transformed_ul, squared=True)

        # auxillary distances to DIFFERENT CLASS points
        distance_matrix_aux = np.multiply(distance_matrix, self.dif_class_matrix)
        distance_matrix_aux[distance_matrix_aux == 0] = 10 * np.amax(distance_matrix_aux)

        # calculate jacobian elementwise
        jac = np.zeros((self.d, self.d))
        jac_ul = np.zeros((self.d, self.d))

        # labeled part
        # most stupid loops ever
        for i in range(self.m):
            for j in range(self.k):
                # index of k-th nearest same class neighbour of i
                index_j = self.eta_index[i, j]
                # reference distance to deciding if there is a push-step derivative at all
                reference_distance = distance_matrix[i, index_j] + 1
                # find possible impostors
                impostors = [z for z, x in enumerate(distance_matrix_aux[i, ] < reference_distance) if x]
                # if there are impostors for given pair x_i and x_j
                if np.size(impostors) > 0:
                    for imp in impostors:
                        p1 = 2 * np.dot((X_transformed[i, ].T - X_transformed[index_j, ].T), np.reshape((self.X_labeled[i, ] - self.X_labeled[index_j, ]), (1, self.d)))
                        p2 = 2 * np.dot((X_transformed[i, ].T - X_transformed[imp, ].T), np.reshape((self.X_labeled[i, ] - self.X_labeled[imp, ]), (1, self.d)))
                        jac += (1 - self.omega[0]) * (p1 - p2)
                    jac += self.omega[0] * 2 * np.dot((X_transformed[i, ].T - X_transformed[index_j, ].T), np.reshape((self.X_labeled[i, ] - self.X_labeled[index_j, ]), (1, self.d)))
                # if there is NO impostors for given pair x_i and x_j
                else:
                    jac += self.omega[0] * 2 * np.dot((X_transformed[i, ].T - X_transformed[index_j, ].T), np.reshape((self.X_labeled[i, ] - self.X_labeled[index_j, ]), (1, self.d)))

        # unlabeled part
        for i in range(self.m):
            for j in range(self.k):
                # index of k-th nearest same class neighbour of i
                index_j_ul = self.eta_index_ul[i, j]
                # reference distance to deciding if there is a push-step derivative at all
                reference_distance_ul = distance_matrix_ul[i, index_j_ul] + 1
                # find possible impostors
                impostors_ul = [z for z, x in enumerate(distance_matrix_aux[i, ] < reference_distance_ul) if x]
                # if there are impostors for given pair x_i and x_j
                if np.size(impostors_ul) > 0:
                    for imp in impostors_ul:
                        p1 = 2 * np.dot((X_transformed[i, ].T - X_transformed_ul[index_j_ul, ].T),
                                         np.reshape((self.X_labeled[i, ] - self.X_unlabeled[index_j_ul, ]), (1, self.d)))
                        p2 = 2 * np.dot((X_transformed[i, ].T - X_transformed[imp, ].T),
                                         np.reshape((self.X_labeled[i, ] - self.X_labeled[imp, ]), (1, self.d)))
                        jac_ul += (1 - self.omega[1]) * (p1 - p2)
                    jac_ul += self.omega[1] * 2 * np.dot((X_transformed[i, ].T - X_transformed_ul[index_j_ul, ].T),
                                                    np.reshape((self.X_labeled[i, ] - self.X_unlabeled[index_j_ul, ]), (1, self.d)))
                # if there is NO impostors for given pair x_i and x_j
                else:
                    jac_ul += self.omega[1] * 2 * np.dot((X_transformed[i, ].T - X_transformed_ul[index_j_ul, ].T),
                                                    np.reshape((self.X_labeled[i, ] - self.X_unlabeled[index_j_ul, ]), (1, self.d)))

        jac_total = self.omega[2] * jac + (1 - self.omega[2]) * jac_ul
        #jac_total = np.reshape(jac_total, (1, self.d ** 2))

        return jac_total

    def score(self, y_true, y_predicted):

        return accuracy_score(y_true, y_predicted)

    def predict_proba(self, X):

        # Check if fit had been called
        check_is_fitted(self, ['X_labeled', 'X_unlabeled', 'y'])
        probabilities = super(SemiSupervisedLargeMarginNearestNeighbor, self).predict_proba(self.transform(X))

        return probabilities
