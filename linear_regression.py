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

        self.coefs_ = np.linalg.pinv(np.hstack((np.ones((X.shape[0], 1)), X)) if self.include_intercept_ else X) @ y
        # Step 5: Mark model as fitted
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

        y_pred = (np.hstack((np.ones((X.shape[0], 1)), X))) @ self.coefs_ if self.include_intercept_ else X @ self.coefs_
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
        # Step 1: Compute the predicted responses
        y_pred = self.predict(X)
        # Step 2: Compute the loss
        mse = np.mean((y - y_pred) ** 2)
        return mse

    ## addition to the API by me
    def var(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under **variance (VAR) loss function**

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        var : float
            Performance under variance loss function
        """
        if not self.fitted_:
            raise ValueError("Estimator has not been fitted yet")

        if X.size == 0 or y.size == 0:
            return np.nan
        # Step 1: Compute the predicted responses
        y_pred = self.predict(X)
        # Step 2: Compute the loss
        var = np.var(y - y_pred)
        return var
