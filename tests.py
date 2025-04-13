import numpy as np
import sys
import argparse


# === Replace with your actual SVD function ===
from ex1 import svd as your_svd_func


def is_orthonormal(mat, tol=1e-6):
    I = np.eye(mat.shape[1])
    return np.allclose(mat.T @ mat, I, atol=tol)


def run_svd_test(verbose=False):
    num_tests = 50
    print("\nRunning SVD tests...")
    passed = 0

    for test_id in range(num_tests):
        m = np.random.randint(4, 10)
        d = np.random.randint(4, 10)
        r = np.random.randint(1, min(m, d))

        A = np.random.randn(m, r)
        B = np.random.randn(d, r)
        M = A @ B.T  # rank ≤ r

        try:
            U, Sigma, VT = your_svd_func(M)
        except Exception as e:
            print(f"❌ Test {test_id}: Exception: {e}")
            continue

        if U.shape != (m, r) or Sigma.shape != (r, r) or VT.shape != (r, d):
            print(f"❌ Test {test_id}: Shape mismatch")
            print(f"  M.shape = {M.shape}, Expected U: ({m}, {r}), Sigma: ({r}, {r}), VT: ({r}, {d})")
            continue

        M_hat = U @ Sigma @ VT
        if not np.allclose(M, M_hat, atol=1e-6):
            print(f"❌ Test {test_id}: Reconstruction failed (Frobenius error: {np.linalg.norm(M - M_hat):.2e})")
            continue

        if not is_orthonormal(U):
            print(f"❌ Test {test_id}: U not orthonormal")
            continue

        if not is_orthonormal(VT.T):
            print(f"❌ Test {test_id}: V not orthonormal")
            continue

        sigma_yours = np.diag(Sigma)
        _, sigma_np, _ = np.linalg.svd(M, full_matrices=False)
        if not np.allclose(np.sort(sigma_yours), np.sort(sigma_np[:r]), atol=1e-6):
            print(f"❌ Test {test_id}: Singular values mismatch")
            continue

        if verbose:
            print(f"✅ Test {test_id} passed")
        passed += 1

    print("-----------------------")
    print(f"✅ SVD: {passed}/{num_tests} passed ✅")
    print("-----------------------")


# === Stub test types for illustration ===

def run_some_other1(verbose=False):
    print("Running test: some_other1 (stub)")
    print("✅ all stub tests passed\n")


def run_some_other2(verbose=False):
    print("Running test: some_other2 (stub)")
    print("✅ all stub tests passed\n")


# === Test registry ===

TESTS = {
    "svd": run_svd_test,
    # "some_other1": run_some_other1,
    # "some_other2": run_some_other2
}


# === Main script ===

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run selected tests")
    parser.add_argument("tests", nargs="*", help="List of tests to run (e.g., svd some_other1)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    selected_tests = args.tests if args.tests else TESTS.keys()
    verbose = args.verbose

    for name in selected_tests:
        test_func = TESTS.get(name)
        if test_func is None:
            print(f"❌ Unknown test type: '{name}'")
        else:
            test_func(verbose=verbose)

