import sys
import math
import numpy as np


class Chebyshev(object):
    """
    Chebyshev kernel class, containing all functions required to compute the chebyshev kernel fucntion.

    Attributes
    ----------
    order : int
        The polynomial order of orthogonal kernel.
    name : str , default is Chebyshev
        Kernel's name.
    form : Char
        Choose how the Chebyshev kernel to be used wether explicit or recursive.
        Only is applicable to Chebyshev kernel, for other kernels if passed, will be ignored.

    Methods
    -------
    GTn(x, n)
        Vectorized method of calculating chebsyhev polynomials terms.
    kernel(x, y)
        Kernel fucntion to calculate the cheybshev kernel.
    """
    def __init__(self, order, form):
        """
       Constructor for Chebyshev class. 

       Parameters
       ----------
       order : int
            Orthogonal kernel's order.
       form : Char
            The explicit or recursive form. 
            Determines which Chebyshev kernel function to be used whether explicit or recursive.
            Only is applicable to Chebyshev kernel, for other kernels even if passed, will be ignored.
    """
        self.order = order
        self.name = "Chebyshev"
        self.form = form


    def GTn(self, x, n):
        """
        Compute the chebsyhev polynomial terms. 
        GTn refers to Generalized form of nth Term of the chebyshev polynomial. 
        By Generalized, we mean the vectorial multiplication instead of elementwise.

        Parameters
        ----------
         x: float
            A point in dataset.
         n: int
            Order.

        Returns
        -------
        float
            Polynomials term.
        """
        if self.form == "e":
            """
                Explicit form
            """
            return np.cos(n * np.arccos(x))

        elif self.form == "r":
            """
                Recursive form
            """
            try:
                if n == 0:
                    return 1
                elif n == 1:
                    return x
                elif n >= 2:
                    return 2 * x * np.transpose(self.GTn(x, n - 1)) - self.GTn(x, n - 2)
            except:
                sys.exit("order must be equal or grater then 0")

    def kernel(self, x, y):
        """
        Compute the cheybshev kernel for two given vectors.

        Parameters
        ----------
        x: float
            A point in dataset.
        y: float
            A point in dataset.

        Return
        -------
        float
            Calculated kernel for x , y.
        """
        try:
            d = len(x)
        except:
            sys.exit("input data kernel dimension has to be > 1!")

        chebyshev_result = 0
        for i in range(self.order):
            chebyshev_result += (np.inner(self.GTn(x, i), self.GTn(y, i)))
        return chebyshev_result / (np.sqrt((d - np.inner(x, y))) + 0.002)  # +0.002 according to Meng Tian et.al


class Legendre(object):
    """
    Legendre kernel class, containing all functions required to compute the Legendre kernel fucntion.

    Attributes
    ----------
    order : int
        Orthogonal kernel's order.
    name : str , default is Legendre
        Kernel name.

    Methods
    -------
    LegendreTerm(x, n)
        Calculating Legendre polynomials terms.
    kernel(self, x, z)
        Kernel function to calculate the Legendre kernel.
    """
    def __init__(self, order=1):
        """
       Constructor for Legendre class

       Parameters
       ----------
       order : int
           Orthogonal kernel's order.
        """
        self.order = order
        self.name = "Legendre"

    def LegendreTerm(self, x, n):
        """
        Compute the Legendre polynomials terms.

        Parameters
        ----------
        x: float
          A point in dataset.
        n: int
            Order.

       Returns
       -------
       float
          computed polynomial term of order n for the given x.
        """
        try:
            if n > 1:
                return ((2 * n - 1) * (x * self.LegendreTerm(x, n - 1)) - (n - 1) * self.LegendreTerm(x, n - 2)) / n
            elif n == 1:
                return x
            elif n == 0:
                return 1
            return 0
        except:
            sys.exit("order must be equal or grater then 0")

    def kernel(self, x, y):
        """
        Compute the Legendre kernel for two given data points.

        Parameters
        ----------
        x: float
            A point in dataset.
        y: float
            A point in dataset.

       Returns
       -------
       float
            Calculated kernel for x , y.
        """
        result = 1
        temp = 0
        for i in range(self.order):
            temp += self.LegendreTerm(x, i) * self.LegendreTerm(y, i)
        for t in temp:
            result += result * t

        return result


class Gegenbauer(object):
    """
    Gegenbauer kernel class,containing all functions required to compute the Gegenbauer kernel fucntion.

    Attributes
    ----------
    order : int
        Orthogonal kernel's order.
    Lambda : float
        Hyperparameter of Gegenbauer kernel.
        lambda should be greater than -0.5.
    name : str , default is Gegenbauer
        Kernel name.

    Methods
    -------
    GegenbauerTerm(x, n)
        Used in Gegenbauer kernel function that calculates the gegenbauer polynomial terms.
    GegenbauerWeight(x, z)
        Used in Gegenbauer kernel to calculate gegenbauer relevant weight.
    GegenbauerScale(n)
        Used in Gegenbauer kernel which is accounted to overcome the computational difficulties such as annihilation effect .
    kernel(x, z)
         Kernel fucntion to calculate the cheybshev kenrel fucntion.
    """

    def __init__(self, order=1, Lambda=1):
        """
        Constructor for Gegenbauer class

        Parameters
        ----------
        order : int
            Orthogonal kernel's order.
        Lambda : float
            Hyperparameter of Gegenbauer kernel.
            lambda should be greater than -0.5.
        """
        self.order = order
        self.Lambda = Lambda
        self.name = "Gegenbauer"

    def GegenbauerTerm(self, x, n):
        """
        Use in Gegenbauer kernel to calculate the gegenbauer polynomials.

        Parameters
        ----------
        x: float
          A point in dataset.
        n: int
            Order.

       Returns
       -------
       float
          Polynomials term.
        """

        try:
            if n == 0:
                return 1
            elif n == 1:
                return 2 * self.Lambda * x
            elif n > 1:
                return ((2 * (n - 1 + self.Lambda) / n * x * self.GegenbauerTerm(x, n - 1)) - ( ((n + (2 * self.Lambda) - 2) / n) * self.GegenbauerTerm(x, n - 2)))
            return 0
        except:
            sys.exit("order must be equal or grater then 0")

    def GegenbauerWeight(self, x, y):
        """
       The function to calculate gegenbauer relevant weight.

       Parameters
       ----------
       x: float
          A point in dataset.
       y: float
          A point in dataset.

      Returns
      -------
      float
          Gegenbauer weight.
     """
        return ((1 - x ** 2) * (1 - y ** 2)) ** (self.Lambda - 0.5)

    def GegenbauerScale(self, n):
        """
        The scale fucntion to prevent computational difficulties of the Gegenbauer kernel.

        Parameters
        ----------
        n: int

        Return
        -------
        float
           Gegenbauer scaling amount.
        """

        return (math.sqrt(n + 1) * abs(self.GegenbauerTerm(1, n))) ** -1

    def kernel(self, x, y):
        """
        The function to compute the Gegenbauer kernel.

        Parameters
        ----------
        x: float
            A point in dataset.
        y: float
            A point in dataset.
       Return
       -------
       float
           Calculated kernel for x , y.
     """
        result = 1
        temp = 0
        for i in range(self.order):
            temp += self.GegenbauerTerm(x, i) * self.GegenbauerTerm(y, i) * self.GegenbauerWeight(x, y) * self.GegenbauerScale(i) ** 2
        for t in temp:
            result += result * t
        return result


class Jacobi(object):
    """
    Jacobi kernel class, containing all functions required to compute the Jacobi kernel fucntion.
    For sake of simplicity and acording to the most well-known resources, teh main computation process has been 
    splitted to multiple functions, the An(n),Bn(n),Cn(n) functions
     are directly used in JacobiTerm fucntion.

    Attributes
    ----------
    psi : float
        Hyperparameter of Jacobi kernel.
        psi should be greater than -1.
    omega : float
         Hyperparameter of Jacobi kernel.
         omega should be greater than -1.
    order : int
        Orthogonal kernel's order.
    noise : float
        A noise  only applicable to weight function of Jacobi kernel. recommended values can be 0.1, 0.01,...
    name : str , default is Jacobi
        Kernel name.

    Methods
    -------
    An(n)
        Use in JacobiTerm for calculating Jacobi Jacobi polynomilas.
    Bn(n)
        Use in JacobiTerm for calculating Jacobi Jacobi polynomilas.
    Cn(n)
        Use in JacobiTerm for calculating Jacobi Jacobi polynomilas.
    JacobiTerm(n, x)
        Use in kernel to calculate the jacobi polynimials.
    weight(x, z)
        Use in kernel to calculate the Jacobi kernel weight.
    kernel(x, z)
        Jacobi kernel function
    """


    def __init__(self, psi, omega, order=3, noise=0.1):
        """
        Constructor for Jacobi class.

        Attributes
        ----------
        psi : float
            Hyperparameter of Jacobi kernel.
            psi should be greater than -1.
        omega : float
             Hyperparameter of Jacobi kernel.
             omega should be greater than -1.
        order : int
            Orthogonal kernel's order.
        noise : float
            A noise  only applicable to weight function of Jacobi kernel. recommended values can be 0.1, 0.01,...
        """
        self.psi = psi
        self.omega = omega
        self.order = order
        self.noise = noise
        self.name = "Jacobi"

    def An(self, n):
        """
       Used in JacobiTerm for calculating Jacobi Jacobi polynomilas.

       Parameters
       ----------
       n: int

      Returns
      -------
      float
          An.
     """

        return ((2 * n + self.psi + self.omega + 1) * (2 * n + self.psi + self.omega + 2)) / (2 * (n + 1) * (n + self.psi + self.omega + 1))

    def Bn(self, n):
        """
        Used in JacobiTerm for calculating Jacobi Jacobi polynomilas.

        Parameters
        ----------
        n: int

        Return
        -------
        float
            Bn.
        """
        return (((self.omega ** 2) - (self.psi ** 2)) * (2 * n + self.psi + self.omega + 1)) / ( 2 * (n + 1) * (n + self.psi + self.omega + 1) * ((2 * n) + self.psi + self.omega))

    def Cn(self, n):
        """
        Use in JacobiTerm for calculating Jacobi Jacobi polynomilas.

        Parameters
        ----------
        n: int

        Return
        -------
        float
            Cn.
        """
        return ((n + self.psi) * (n + self.omega) * (2 * n + self.psi + self.omega + 2)) / ((n + 1) * (n + self.psi + self.omega + 1) * (2 * n + self.psi + self.omega))

    def JacobiTerm(self, n, x):
        """
        Used in Jacobi kernel to calculate the Jacobi polynomials.

        Parameters
        ----------
        x: float
           A point in dataset.
        n: int
            Order.

        Return
        -------
        float
            Jacobi polynomials term.
      """
        if n == 0:
            return 1
        if n == 1:
            return 0.5 * np.dot((self.psi + self.omega + 2), x) + 0.5 * (self.psi - self.omega)
        else:
            return ((np.dot(self.An(n - 1), x) + self.Bn(n - 1)) * self.JacobiTerm(n - 1, x)) - (self.Cn(n - 1) * self.JacobiTerm(n - 2, x))

    def Weight(self, x, y):
        """
        Use in Jacobi kernel to calculate Jacobi relevant weight.

        Parameters
        ----------
        x: float
            A point in dataset.
        y: float
            A point in dataset.

        Return
        -------
        float
           Jacobi weight.
        """
        d = len(x)
        return (((d - np.dot(x, y)) + self.noise) ** self.psi) * (((d + np.dot(x, y)) + self.noise) ** self.omega)

    def kernel(self, x, y):
        """
        Calculate the Gegenbauer kenrel fucntion.

        Parameters
        ----------
        x: float
          A point in dataset.
        y: float
          A point in dataset.
        Return
        -------
        float
            Calculated kernel for x , y.
        """
        _sum = 0
        result = 1

        try:
            d = len(x)
        except:
            sys.exit("input data kernel dimension has to be > 1!")
        for i in range(self.order + 1):
            _sum += self.JacobiTerm(i, x) * np.transpose(self.JacobiTerm(i, y))
        for i in _sum * self.Weight(x, y):
            result = result * i

        return result
