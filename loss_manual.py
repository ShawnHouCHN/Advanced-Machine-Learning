import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances
# k-value here
k = 4

dataset = load_iris()
X, y = dataset.data, dataset.target

# number of observations
m = X.shape[0]

# initial L - just identity
L_init = np.eye(X.shape[1])
L = L_init

# calculate linear transformation X L'
X_transformed = np.dot(np.asmatrix(X),np.asmatrix(L).T)

# calculate pairwise distances between all samples in transformed X
distance_matrix = euclidean_distances(X_transformed,squared=True)

# find k nearest SAME CLASS elements
# non-zero elements in each row correspond to closest nearest SAME CLASS neighbours
eta = np.zeros([m,m])
for i in range(0,m-1):
    same_class = y == y[i]
    closest = distance_matrix[i,same_class].argsort()[1:k+1] # excluding self obviously
    eta[i,closest] = 1

# more efficient way?

# find all DIFFERENT class points
# non-zero elements in each row correspond to different class points
diff_class = np.zeros([m,m])
for i in range(0,m-1):
    different_class = y != y[i]
    diff_class[i,different_class] = 1

# more efficient way?

# LMNN loss function
omega_1 = 0.5

# PULL - step
pull_sum = omega_1 * np.sum(np.multiply(eta,distance_matrix))

# PUSH - step
diff_class_dist = np.multiply(diff_class,distance_matrix)
# select k distances to closest same class neighbours
same_class_dist = np.sort(np.multiply(eta,distance_matrix))[:,-k:]
# arbitrary + 1 increase
same_class_dist += 1

push_sum = 0
# loop for k of each neighbours
for i in range(k):
    # loop for each sample
    for j in range(m):
        # pick 1 distance - k-th closest element to m-th observation
        reference_value = same_class_dist[j,i]
        # find which different class distances are maller than picked up above
        relevant_diff_class_dist = diff_class_dist[j,(diff_class_dist < reference_value)[0,:]]
        # throw out all 0-s
        relevant_diff_class_dist = relevant_diff_class_dist[relevant_diff_class_dist != 0]
        # add the difference to total push-sum
        push_sum += np.sum(reference_value - relevant_diff_class_dist)
push_sum = (1 - omega_1) * push_sum

# total loss value here
total_loss = pull_sum + push_sum

# gradient with respect to each elements 