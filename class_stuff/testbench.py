from LMNN import LMNN_C
from sklearn.datasets import load_iris

# data
dataset = load_iris()
X, y = dataset.data, dataset.target

nn = LMNN_C()

a, b = nn.LMNN_fit(X, y)

print(nn.loss_simple(b))
# print(nn.loss_simple_jac(b))

# from loss_manual_v2 - Kopi.py -> 339.295890934583
# from testbench.py and LMNN.py -> 339.295890934583
#
# that means so far so good xD
