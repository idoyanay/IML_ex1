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


def test_svd(your_svd_func, num_tests=50, tol=1e-6):
    def is_orthonormal(mat):
        I = np.eye(mat.shape[1])
        return np.allclose(mat.T @ mat, I, atol=tol)

    for test_id in range(num_tests):
        print(f"\n--- Test {test_id + 1} ---")
        m = np.random.randint(4, 10)
        d = np.random.randint(4, 10)
        r = np.random.randint(1, min(m, d))

        # Create a matrix of rank r
        A = np.random.randn(m, r)
        B = np.random.randn(d, r)
        M = A @ B.T  # Shape: m x d, Rank ≤ r

        try:
            U, Sigma, VT = your_svd_func(M)
        except Exception as e:
            print(f"❌ Exception in your SVD function: {e}")
            continue

        # Validate shapes
        expected_U_shape = (m, r)
        expected_Sigma_shape = (r, r)
        expected_VT_shape = (r, d)

        if U.shape != expected_U_shape or Sigma.shape != expected_Sigma_shape or VT.shape != expected_VT_shape:
            print("❌ Output shape mismatch:")
            print(f"  M.shape = {M.shape}")
            print(f"  U.shape = {U.shape}, expected {expected_U_shape}")
            print(f"  Sigma.shape = {Sigma.shape}, expected {expected_Sigma_shape}")
            print(f"  VT.shape = {VT.shape}, expected {expected_VT_shape}")
            continue

        # Reconstruct M
        M_hat = U @ Sigma @ VT

        if not np.allclose(M, M_hat, atol=tol):
            error = np.linalg.norm(M - M_hat)
            print(f"❌ Reconstruction error too high.")
            print(f"  Frobenius norm of difference: {error:.2e}")
            print(f"  Original M:\n{M}")
            print(f"  Reconstructed M_hat:\n{M_hat}")
            continue

        # Check orthonormality
        if not is_orthonormal(U):
            print("❌ U is not orthonormal.")
            print(f"  U.T @ U:\n{U.T @ U}")
            continue

        if not is_orthonormal(VT.T):
            print("❌ V is not orthonormal.")
            print(f"  V.T @ V:\n{VT.T.T @ VT.T}")
            continue

        # Compare singular values
        sigma_from_yours = np.diag(Sigma)
        _, sigma_np, _ = np.linalg.svd(M, full_matrices=False)
        if not np.allclose(sorted(sigma_from_yours), sorted(sigma_np[:r]), atol=tol):
            print("❌ Singular values mismatch.")
            print(f"  Yours:   {np.sort(sigma_from_yours)}")
            print(f"  NumPy's: {np.sort(sigma_np[:r])}")
            continue

        print("✅ Passed.")

    print("\nAll tests done.")


def main():
    
    test_svd(svd, num_tests=50, tol=1e-6)







if __name__ == "__main__":

    main()