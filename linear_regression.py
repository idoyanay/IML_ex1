import numpy as np
from typing import NoReturn


class LinearRegression:
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem

    Attributes
    ----------
    fitted_ : bool
        Indicates if estimator has been fitted. Set to True in ``self.fit`` function

    include_intercept_: bool
        Should fitted model include an intercept or not

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by linear regression. To be set in
        `LinearRegression.fit` function.
    """

    def __init__(self, include_intercept: bool = True):
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """

        self.fitted_ = False
        self.include_intercept_ = include_intercept
        self.coefs_ = None


    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """

        # Determine the number of features, including intercept if applicable
        if self.include_intercept_:
            X_intercept = np.ones((X.shape[0], X.shape[1] + 1))
            X_intercept[:, 1:] = X
            X = X_intercept

        M = X.shape[0]
        D = X.shape[1]

        # Step 1: Compute the SVD of the design matrix
        U, Sigma, V = self._svd(X)
        # Step 2: Compute the pseudo-inverse of Sigma
        Sigma_inv = np.zeros((D, M))
        for i in range(Sigma.shape[0]):
            Sigma_inv[i, i] = 1 / Sigma[i, i]
        # Step 3: Compute the pseudo-inverse of the design matrix
        X_pseudo_inv = V.T @ Sigma_inv @ U.T
        # Step 4: Compute the coefficients
        self.coefs_ = X_pseudo_inv @ y
        # Set fitted_ to True
        self.fitted_ = True


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if not self.fitted_:
            raise ValueError("Estimator has not been fitted yet")
        if self.include_intercept_:
            X_intercept = np.ones((X.shape[0], X.shape[1] + 1))
            X_intercept[:, 1:] = X
            X = X_intercept
        # Step 1: Compute the predicted responses
        y_pred = X @ self.coefs_
        return y_pred

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under **mean squared error (MSE) loss function**

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        if not self.fitted_:
            raise ValueError("Estimator has not been fitted yet")
        # Step 1: Compute the predicted responses
        y_pred = self.predict(X)
        # Step 2: Compute the loss
        loss = np.mean((y - y_pred) ** 2)
        return loss

    def _svd(M):
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
