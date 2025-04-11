import numpy as np
import pandas as pd 

def main():

    # Step 1: Create a sample matrix
    M = np.random.randint(0, 5, (3,4))

    # Step 2: Compute M^T * M and M * M^T
    MTM = M.T @ M
    MMT = M @ M.T

    # Step 3: Compute eigenvalues and eigenvectors
    eigvals_V, V = np.linalg.eigh(MTM)  # Right singular vectors
    eigvals_U, U = np.linalg.eigh(MMT)  # Left singular vectors
    print("eigvals_V:\n", eigvals_V)
    print("eigvals_U:\n", eigvals_U)

    # Step 4: Sort eigenvalues (and corresponding vectors) in descending order
    idx_V = np.argsort(eigvals_V)[::-1]
    idx_U = np.argsort(eigvals_U)[::-1]
    V = V[:, idx_V]
    U = U[:, idx_U]
    singular_vals = np.sqrt(np.sort(eigvals_V)[::-1])  # Sorted singular values

    # Step 5: Construct Sigma
    Sigma = np.zeros_like(M, dtype=float)
    np.fill_diagonal(Sigma, singular_vals)

    # Reconstruct M to check
    M_reconstructed = U @ Sigma @ V.T
    error = np.linalg.norm(M - M_reconstructed)

    # print("Original M:\n", M)
    # print("Reconstructed M:\n", M_reconstructed)
    # print("Reconstruction error:", error)
    pass




if __name__ == "__main__":
    main()