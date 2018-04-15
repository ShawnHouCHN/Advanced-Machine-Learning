import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import gen_batches
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y, check_random_state
from sklearn.metrics import accuracy_score

class LargeMarginNearestNeighbor(KNeighborsClassifier):

    def __init__(self, L=None, n_neighbors=4, method='L-BFGS-B', options={'disp': True}, max_iter=200, tol=1e-5, omega=0.5):
        
        """Largest Margin Nearest Neighbor. Train a transformation on X to fit in KNN
        Parameters
        ----------
        omega: weight parameter
        """
        
        super(LargeMarginNearestNeighbor, self).__init__(n_neighbors=n_neighbors)
        # Number of neighbors
        self.k = n_neighbors

        self.method = method
        self.options = options
        self.options['maxiter'] = max_iter
        self.max_iter = max_iter
        
        
        self.L_init = L
        
        self.tol=tol
        self.omega = omega
        # Defining the "global" variables in the class, some will hopefully disapear during code optimization
        self.X = None
        self.y = None
        self.m = None
        self.d = None
        self.dif_class_matrix = None
        self.eta_index = None
        self.eta = None

    def fit(self, X, y):

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
        
        #res = minimize(fun=self.loss_gradient, x0=self.L_init, method=self.method,
        #               jac=self.loss_simple_jac, options=self.options)
        
        L, loss, details = fmin_l_bfgs_b(func=self.loss_gradient, x0=self.L_init, bounds=None,
                                                  m=100, pgtol=self.tol, maxfun=500*self.max_iter,
                                                  maxiter=self.max_iter, disp=4, callback=None)
        
        L_optim = L.reshape((self.d, self.d))

        #L_optim_flat = np.reshape(L_optim, (self.d**2, ))

        self.L_= L_optim
        # alternative PCA  # Why the PCA?
        # pca = PCA(n_components=d)
        # a = pca.fit(X)
        # L_init_alt = a.components_
        # L_init_alt = np.reshape(L_init_alt, (d ** 2, ))
        # print(self.loss_simple(L_init_alt))

        # print(self.loss_simple(L_optim_flat))
        # print(self.loss_simple_jac(L_optim_flat))
        
        # Fit a simple nearest neighbor classifier with the learned metric
        super(LargeMarginNearestNeighbor, self).fit(self.transform(), y)   
        
        return self
    
    def predict(self, X):
       
        # Check if fit had been called
        check_is_fitted(self, ['X', 'y'])
        y_pred = super(LargeMarginNearestNeighbor, self).predict(self.transform(X))

        return y_pred
    
    
    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
    
    def transform(self, X=None):
        """Applies the learned transformation to the inputs.

        Parameters
        ----------
        X : array_like
            An array of data samples with shape (n_samples, n_features_in) (default: None, defined when fit is called).

        Returns
        -------
        array_like
            An array of transformed data samples with shape (n_samples, n_features_out).

        """
        if X is None:
            X = self.X
        else:
            X = check_array(X)

        return X.dot(self.L_.T)
    
    
    ################################################# Xiaoshen Implemntation ############################################################
    def loss_gradient(self, L_in):

        # transform L from vector back to matrix
        L = np.reshape(L_in,(self.d, self.d))
        # calculate linear transformation X L'
        X_transformed = np.dot(np.asmatrix(self.X), np.asmatrix(L).T)

        # calculate pairwise distances between all samples in transformed X
        distance_matrix = euclidean_distances(X_transformed, squared=True)

        # LMNN loss function
        omega_1 = self.omega
        
        # calculate jacobian elementwise
        jac = np.zeros((self.d, self.d))

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
                ##impostor_num += sum(distance_matrix_aux[i, ] <= reference_distance)  ##Valentine
                ##impostors = [j for j, x in enumerate(distance_matrix_aux[i, ] <= reference_distance) if x]  ##Valentine
                impostors, = np.where((distance_matrix_aux[i, ] - reference_distance)<= 0 )
                impostor_num += np.size(impostors)
                i_indexes = np.append(i_indexes, np.repeat(i, np.size(impostors)))
                j_indexes = np.append(j_indexes, np.repeat(j, np.size(impostors)))
                impostor_indexes = np.append(impostor_indexes, impostors)
                
                if j < self.k: 
                    index_j = self.eta_index[i, j]
                    reference_distance = distance_matrix[i,index_j] + 1
                    # find possible impostors
                    impostors, = np.where((distance_matrix_aux[i, ] - reference_distance)< 0 )
                    if np.size(impostors) > 0:
                        for imp in impostors:
                            p1 = 2 * np.dot((X_transformed[i, ].T - X_transformed[index_j, ].T), np.reshape((self.X[i, ] - self.X[index_j, ]), (1, self.d)))
                            p2 = 2 * np.dot((X_transformed[i, ].T - X_transformed[imp, ].T), np.reshape((self.X[i, ] - self.X[imp, ]), (1, self.d)))
                            jac += (1 - omega_1) * (p1 - p2)
                        jac += omega_1 * 2 * np.dot((X_transformed[i, ].T - X_transformed[index_j, ].T), np.reshape((self.X[i, ] - self.X[index_j, ]), (1, self.d)))
                    # if there is NO impostors for given pair x_i and x_j
                    else:
                        jac += omega_1 * 2 * np.dot((X_transformed[i, ].T - X_transformed[index_j, ].T), np.reshape((self.X[i, ] - self.X[index_j, ]), (1, self.d)))

        #jac = np.reshape(jac, (1, self.d**2))                
                
        i_indexes = i_indexes.astype(int)
        j_indexes = j_indexes.astype(int)
        impostor_indexes = impostor_indexes.astype(int)
        push_sum = 0
        for n in range(impostor_num):
            i = i_indexes[n]
            j = j_indexes[n]
            l = impostor_indexes[n]
            push_sum += (1 + distance_matrix[i, j] - distance_matrix[i, l])
        push_sum = (1 - omega_1) * push_sum
        total_loss = push_sum + pull_sum   
        
        
        return total_loss, L.dot(jac).flatten()
   ################################################################################################################################################

    def loss_simple(self, L_in):

        # transform L from vector back to matrix
        L = np.reshape(L_in,(self.d, self.d))
        # calculate linear transformation X L'
        X_transformed = np.dot(np.asmatrix(self.X), np.asmatrix(L).T)

        # calculate pairwise distances between all samples in transformed X
        distance_matrix = euclidean_distances(X_transformed, squared=True)

        # LMNN loss function
        omega_1 = self.omega

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


        push_sum = (1 - omega_1) * push_sum

        total_loss = push_sum + pull_sum

        return total_loss
    

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
        omega_1 = self.omega

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
