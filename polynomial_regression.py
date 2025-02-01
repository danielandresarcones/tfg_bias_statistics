import numpy as np
from numpy.polynomial.polynomial import Polynomial

class PolynomialRegression:
    """
    A class to represent a polynomial regression model.

    Attributes
    ----------
    degree : int
        The degree of the polynomial regression model.
    coefficients : ndarray
        The coefficients of the polynomial regression model.

    Methods
    -------
    evaluate(x)
        Evaluates the polynomial regression model at a given input vector x.
    __repr__()
        Returns a string representation of the PolynomialRegression instance.
    __call__(x)
        Allows the instance to be called as a function to evaluate the polynomial regression model at a given input vector x.
    """

    def __init__(self, degree):
        self.degree = degree
        self.poly = Polynomial([0] * (degree + 1))
        self.coefficients = None

    def evaluate(self, x):
        if self.coefficients is None:
            raise ValueError("The model has not been fitted yet.")
        return np.polyval(self.coefficients[::-1], x)

    def __repr__(self):
        return f"PolynomialRegression(degree={self.degree}, coefficients={self.coefficients})"

    def __call__(self, x):
        return self.evaluate(x)
