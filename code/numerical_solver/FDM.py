import numpy as np
import scipy
import scipy.sparse as sp


#need to update to include dirichlet boundary conditions

def solver(F,N, neumann = True):
    h = 1/(N - 1)
    F_num = -F
    # Initialize the sparse matrix A
    main_diagonal = np.ones(N**2) * -4 / h**2
    offset_diagonals = np.ones(N**2 - 1) / h**2
    offset_diagonals[N-1::N] = 0  # Fix the boundary effects where wrap-around occurs
    far_diagonals = np.ones(N**2 - N) / h**2

    # Create the sparse matrix A using diags
    A = sp.diags(
        [main_diagonal, offset_diagonals, offset_diagonals, far_diagonals, far_diagonals],
        [0, -1, 1, -N, N], shape=(N**2, N**2), format='csr')
    if neumann:

        for i in range(N):
            if i == 0 or i == N-1:
                A[i,:] = A[-(i+1),:] = 0
                A[i,i] = A[-(i+1),-(i+1)] = 1/h
            if i % N == 0 or (i+1) % N == 0:
                A[i,:] = 0
                A[i,i] = 1/h

    F_num = F_num.ravel()

    if neumann:
        F_num[0:N] =F_num[-N:] = F_num[::N] = F_num[N-1::N] = 0

    # Solve the linear system
    u = scipy.sparse.linalg.spsolve(A, F_num)
    U = u.reshape((N, N))
    return U