import svddash as cpp
from numpy.linalg import svd
import numpy as np


np.random.seed(1)
d = 5
A = np.random.standard_normal(size=(d, d))

print(svd(A, compute_uv=False))

V1 = np.random.standard_normal(size=(d, d))
threshold = 0.5
alpha = 0.1
f0, df, d2f = cpp.compute(A, threshold, alpha)
dfdV = np.sum(df*V1)
d2fdVdV = np.sum(V1[:, :, None, None]*d2f*V1[None, None, :, :])
err1_old = 1e6
err2_old = 1e6
print("""
################################################################################
###### Perform a taylor test to check the first and second derivative. #########
################################################################################
""")
for i in range(0, 20):
    eps = 0.5 ** (i+5)
    f1 = cpp.compute(A+eps*V1, threshold, alpha)[0]
    err1 = np.abs(f0 + eps*dfdV - f1)
    err2 = np.abs(f0 + eps*dfdV + 0.5*eps*eps*d2fdVdV - f1)
    print(err1, err1/err1_old)
    print(err2, err2/err2_old)
    assert err1/err1_old < 0.26 # assert proper convergence rates
    assert err2/err2_old < 0.13
    err1_old = err1
    err2_old = err2
    if err1 < 1e-13 or err2 < 1e-13:
        break
    print("=============================")
