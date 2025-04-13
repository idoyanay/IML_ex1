import numpy as np
import pandas as pd 


def svd(M):
        # Step 2: Compute M^T * M and M * M^T
    r = np.linalg.matrix_rank(M)
    MTM = M.T @ M
    MMT = M @ M.T

    eigvals_V, V = np.linalg.eigh(MTM)  # Right singular vectors
    eigvals_V = eigvals_V[::-1]  # Sort eigenvalues in descending order
    V = V[:, ::-1]  # Sort eigenvectors accordingly
    V = V.T
    V = V[:r, :]
    
    singular_vals = np.sqrt(eigvals_V)  # Sorted singular values

    # Step 5: Construct Sigma
    
    Sigma = np.zeros((r, r))
    np.fill_diagonal(Sigma, singular_vals)

    

    U = np.zeros((M.shape[0],r))

    for i in range(r):
        sigma_inv = 1/singular_vals[i]
        U_i = sigma_inv * M @ V[i, :]
        U[:, i] = U_i
    
 
    return U, Sigma, V  # Return U, Sigma, and V

def main():
    pass



if __name__ == "__main__":

    main()