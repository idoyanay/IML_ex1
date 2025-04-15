from typing import NoReturn
from linear_regression import LinearRegression
import numpy as np


class PolynomialFitting(LinearRegression):
    """
    Polynomial Fitting using Least Squares estimation
    """
    def __init__(self, k: int):
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        self.k = k
        # assert that k is positive
        assert self.k > 0, "k must be positive"
        super().__init__(include_intercept=False)

    def __check_array(self, array: np.ndarray) -> bool:
        """
        Check if array is 1D or 2D with one column and not empty
        """
        return (array.ndim == 1 or (array.ndim == 2 and array.shape[1] == 1)) and array.size > 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        if self.k >= X.shape[0]:
            print("Warning: k is greater than number of samples, this may lead to overfitting")
        assert self.__check_array(X), f"X must be 1D or 2D with one column and not empty, got {X.ndim}D array with shape {X.shape}"
        assert self.__check_array(y), f"y must be 1D or 2D with one column and not empty, got {y.ndim}D array with shape {y.shape}"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"

        if X.ndim == 2:
            X = X.flatten()



        # Transform the input data
        X_vec = self.__transform(X)
        # Fit the model using the transformed data
        super().fit(X_vec, y)


    # my addition to the class - for override the var method in the LinearRegression class
    def var(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        # assert that X is 1D or 2D with one column
        assert self.__check_array(X), f"X must be 1D or 2D with one column and not empty, got {X.ndim}D array with shape {X.shape}"
        if X.ndim == 2:
            X = X.flatten()
        assert self.__check_array(y), f"y must be 1D or 2D with one column and not empty, got {y.ndim}D array with shape {y.shape}"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"

        # Transform the input data
        X_vec = self.__transform(X)
        # Predict using the transformed data
        return super().var(X_vec, y)
    


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

        assert self.__check_array(X), f"X must be 1D or 2D with one column and not empty, got {X.ndim}D array with shape {X.shape}"
        if X.ndim == 2:
            X = X.flatten()
        # Transform the input data
        X_vec = self.__transform(X)
        # Predict using the transformed data
        return super().predict(X_vec)
        

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

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
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        # Predict using the transformed data

        # the data will transform in the predict method
        return super().loss(X, y)

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        # assert that X is 1D
        return np.vander(X, self.k+1, increasing=True)
