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

# find all DIFFERENT class points
# non-zero elements in each row correspond to different class points
diff_class = np.zeros([m,m])
for i in range(0,m-1):
    different_class = y != y[i]
    diff_class[i,different_class] = 1

# LMNN loss function
omega_1 = 0.5

# PULL - step
pull_sum = omega_1 * np.sum(np.multiply(eta,distance_matrix))

# PUSH - step
diff_class_dist = np.multiply(diff_class,distance_matrix)
same_class_dist = np.multiply(eta,distance_matrix)
same_class_dist[same_class_dist != 0] += 1# arbitrary + 1 value
push_sum = 0

for i in range(k):

















# assign gian values to diagonal so same element will not be closest to itself
test_dist[range(test_dist.shape[0]),range(test_dist.shape[0])] = np.matrix.max(test_dist)*2

# find k nearest SAME CLASS points for each element


# find all DIFFERENT class points
diff_class = np.zeros([m,m])
for i in range(0,m-1):
    different_class = y != y[i]
    diff_class[i,different_class] = 1

# compute loss function in 2 parts
omega = 0.5

# split computations into 2 parts
# part 1
loss_part1 = 0
for i in range(0,m-1):
    for j in range(0,m-1):
        if eta[i,j] != 0:
            # distance between each points with respect to L is already calculated
            loss_part1 += omega * test_dist[i,j]

# part 2
loss_part2 = 0
for i in range(0,m-1):
    for j in range(0,m-1):
        for l in range(0,m-1):
            if eta[i,j] != 0:
                if diff_class[i,l] != 0:
                    # again distances already calculated above
                    loss_part2 += (1-omega) * max(0,
                                                  1 + test_dist[i,j] - test_dist[i,l])

# total loss
Loss = loss_part1 + loss_part2

# next - derivative for d^2 elements of matrix L, just code what I did by hand