'''

Ishigami Function

f(x)= sin(x_1) + a sin^2(x_2) + b x_3^4 sin(x_1)

Description:
Dimensions: 3

The Ishigami function of Ishigami & Homma (1990) is used as an example for uncertainty and sensitivity analysis methods, because it exhibits strong nonlinearity and nonmonotonicity. It also has a peculiar dependence on x3, as described by Sobol' & Levitan (1999).

The values of a and b used by Crestaux et al. (2007) and Marrel et al. (2009) are: a = 7 and b = 0.1. Sobol' & Levitan (1999) use a = 7 and b = 0.05.

'''

import numpy as np
class IshigamiFunction:
    """
    A class to represent the Ishigami function, a well-known test function for uncertainty quantification and sensitivity analysis.

    Attributes
    ----------
    a : float
        The coefficient for the second term of the Ishigami function. Default is 7.
    b : float
        The coefficient for the third term of the Ishigami function. Default is 0.1.

    Methods
    -------
    evaluate(x)
        Evaluates the Ishigami function at a given 3-dimensional input vector x.
    __repr__()
        Returns a string representation of the IshigamiFunction instance.
    __call__(x)
        Allows the instance to be called as a function to evaluate the Ishigami function at a given input vector x.
    """

    def __init__(self, a=7, b=0.1):
        self.a = a
        self.b = b

    def evaluate(self, x):
        if len(x) != 3:
            raise ValueError("Input vector must have exactly 3 dimensions.")
        x1, x2, x3 = x
        return np.sin(x1) + self.a * np.sin(x2)**2 + self.b * x3**4 * np.sin(x1)

    def __repr__(self):
        return f"IshigamiFunction(a={self.a}, b={self.b})"

    def __call__(self, x):
        return self.evaluate(x)