import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier

class LMNN_C(KNeighborsClassifier):

    def __init__(self, L=None, n_neighbors=4, method='L-BFGS-B', options={'disp': True}):

        # Number of neighbors
        self.k = n_neighbors

        self.method = method
        self.options = options

        # initial L - just identity
        # self.L_init = np.eye(d) # moved to be defined in the fit function, because of dependency of data (X, y)
        # self.L_init = np.reshape(L_init, (d**2,))
        self.L_init = L

        # Defining the "global" variables in the class, some will hopefully disapear during code optimization
        self.X = None
        self.y = None
        self.m = None
        self.d = None
        self.dif_class_matrix = None
        self.eta_index = None
        self.eta = None

    def LMNN_fit(self, X, y):

        self.X = X
        self.y = y

        # number of observations
        self.m = self.X.shape[0]

        # number of dimensions
        self.d = self.X.shape[1]

        # initial L - just identity
        if self.L_init is None :
            self.L_init = np.eye(self.d)
        self.L_init = np.reshape(self.L_init, (self.d**2,))

        # initial distances
        initial_distance = euclidean_distances(self.X, squared=True)

        # find indexes of k closest same class points in initial space
        # now saved as sparse matrix as well
        self.dif_class_matrix = np.zeros(shape=(self.m, self.m))
        self.eta_index = np.empty(shape=(self.m, self.k), dtype=int)
        eta_row = np.repeat(range(self.m), self.k)
        eta_col = []
        for i in range(self.m):
            # take points that belong to different class than point i
            dif_class = y != y[i]
            self.dif_class_matrix[i,dif_class] = 1
            # take distances from point i to all other points
            point_distances = initial_distance[i,]
            # assign large values to different class points - so they will never be closest ones
            point_distances[dif_class] = max(point_distances) * 10
            # remember indexes of k closest same class neighbours for point i
            eta_col = np.append(eta_col,point_distances.argsort()[1:self.k + 1])
            self.eta_index[i,] = point_distances.argsort()[1:self.k + 1]
        self.eta = csr_matrix((np.repeat(1, self.m*self.k), (eta_row, eta_col)), shape=(self.m, self.m))

        res = minimize(fun=self.loss_simple, x0=self.L_init, method=self.method,
                       jac=self.loss_simple_jac, options=self.options)
        L_optim = res.x.reshape((self.d, self.d))

        L_optim_flat = np.reshape(L_optim, (self.d**2, ))

        # alternative PCA  # Why the PCA?
        # pca = PCA(n_components=d)
        # a = pca.fit(X)
        # L_init_alt = a.components_
        # L_init_alt = np.reshape(L_init_alt, (d ** 2, ))
        # print(self.loss_simple(L_init_alt))

        # print(self.loss_simple(L_optim_flat))
        # print(self.loss_simple_jac(L_optim_flat))

        return L_optim, L_optim_flat

    def SS_LMNN_fit(selfX, y):
        # Stuff

    def loss_simple(self, L_in):

        # transform L from vector back to matrix
        L = np.reshape(L_in,(self.d, self.d))

        # calculate linear transformation X L'
        X_transformed = np.dot(np.asmatrix(self.X), np.asmatrix(L).T)

        # calculate pairwise distances between all samples in transformed X
        distance_matrix = euclidean_distances(X_transformed, squared=True)

        # LMNN loss function
        omega_1 = 0.5

        # PULL - step
        neighbour_distance_matrix = self.eta.multiply(distance_matrix)
        pull_sum = omega_1 * neighbour_distance_matrix.sum()

        # auxillary distances to DIFFERENT CLASS points
        distance_matrix_aux = np.multiply(distance_matrix,self.dif_class_matrix)
        distance_matrix_aux[distance_matrix_aux == 0] = 10 * np.amax(distance_matrix_aux)

        impostor_num = 0
        i_indexes = []
        j_indexes = []
        impostor_indexes = []
        for i in range(self.m):
            for j in self.eta_index[i,]:
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

    def loss_ssc(self, L_in):
        print("hello")
        # Stuff

    def loss_combined(self,L_in):
        printf("hello")
        # Stuff

    def loss_simple_jac(self, L_in):

        # transform L from vector back to matrix
        L = np.reshape(L_in, (self.d, self.d))

        # calculate linear transformation X L'
        X_transformed = np.dot(np.asmatrix(self.X), np.asmatrix(L).T)

        # find pairwise distances in new space
        distance_matrix = euclidean_distances(X_transformed, squared=True)

        # auxillary distances to DIFFERENT CLASS points
        distance_matrix_aux = np.multiply(distance_matrix, self.dif_class_matrix)
        distance_matrix_aux[distance_matrix_aux == 0] = 10 * np.amax(distance_matrix_aux)

        # calculate jacobian elementwise
        jac = np.zeros((self.d, self.d))

        # omega weight here
        omega_1 = 0.5

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
                        p1 = 2 * np.dot((X_transformed[i, ].T - X_transformed[index_j, ].T), np.reshape((self.X[i, ] - self.X[index_j, ]), (1, self.d)))
                        p2 = 2 * np.dot((X_transformed[i, ].T - X_transformed[imp, ].T), np.reshape((self.X[i, ] - self.X[imp, ]), (1, self.d)))
                        jac += (1 - omega_1) * (p1 - p2)
                    jac += omega_1 * 2 * np.dot((X_transformed[i, ].T - X_transformed[index_j, ].T), np.reshape((self.X[i, ] - self.X[index_j, ]), (1, self.d)))
                # if there is NO impostors for given pair x_i and x_j
                else:
                    jac += omega_1 * 2 * np.dot((X_transformed[i, ].T - X_transformed[index_j, ].T), np.reshape((self.X[i, ] - self.X[index_j, ]), (1, self.d)))

        jac = np.reshape(jac, (1, self.d**2))
        return jac

    def loss_ssc_jac(self, L,in):
        print("hello")
        # Stuff

    def loss_combined_jac
