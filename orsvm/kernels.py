import sys
import math
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

class Chebyshev(object):
    """
    Chebyshev kernel class, containing all functions required to compute the chebyshev kernel function.

    Attributes
    ----------
    order : int
        The polynomial order of orthogonal kernel.
    name : str , default is Chebyshev
        Kernel's name.
    form : Char
        Choose how the Chebyshev kernel to be used whether explicit or recursive.
        Only is applicable to Chebyshev kernel, for other kernels if passed, will be ignored.

    Methods
    -------
    GTn(x, n)
        Vectorized method of calculating chebsyhev polynomials terms.
    kernel(x, y)
        Kernel function to calculate the cheybshev kernel.
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
        By Generalized, we mean the vectorized multiplication instead of elementwise.

        Parameters
        ----------
         x: numpy array / float
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
                sys.exit("order must be equal or greater than 0")
        else :
            logging.error('The parameter form can only be "r" for recursive or "e" for the explicit equation implementation of the chebyshev kernel!')
            sys.exit()

    def kernel(self, x, y):
        """
        Compute the cheybshev kernel for two given vectors.

        Parameters
        ----------
        x: numpy array
            A point in dataset.
        y: numpy array
            A point in dataset.

        Return
        -------
        float
            Calculated kernel for x , y.
        """
        if (self.order < 0) :
            logging.error("order must be equal or greater than 0")
            sys.exit()
        
        
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
    Legendre kernel class, containing all functions required to compute the Legendre kernel function.

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
        x: numpy array / float
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
            sys.exit("order must be equal or greater than 0")

    def kernel(self, x, y):
        """
        Compute the Legendre kernel for two given data points.

        Parameters
        ----------
        x: numpy array
            A point in dataset.
        y: numpy array
            A point in dataset.

       Returns
       -------
       float
            Calculated kernel for x , y.
        """
        if (self.order < 0):
            logging.error("order must be equal or greater than 0")
            sys.exit()

       
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
        x: numpy array / float
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
            sys.exit("order must be equal or greater than 0")

    def GegenbauerWeight(self, x, y):
            """
           The function to calculate gegenbauer relevant weight.
           Parameters
           ----------
           x: numpy array / float
           y: numpy array / float

           Returns
           -------
           float
               Gegenbauer weight.
            """
            if self.Lambda <= -0.5:
                logging.error("** Error: hyperparamter invalid range! For the Gegenbauer kernel function:  -0.5< kernelParameter ")
                sys.exit()
            elif -0.5 <self.Lambda <= 0.5:
                return 1
            elif self.Lambda >= 0.5:
                return ((1 - x ** 2) * (1 - y ** 2)) ** (self.Lambda - 0.5)


    def GegenbauerScale(self, n):
        """
        The scale function to prevent computational difficulties of the Gegenbauer kernel.

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
        x: numpy array
            A point in dataset.
        y: numpy array
            A point in dataset.
       Return
       -------
       float
           Calculated kernel for x , y.
     """
        if (self.order <0) :
            logging.error("order must be equal or greater than 0")
            sys.exit()
        result = 1
        temp = 0
        
        if (self.Lambda is None) :
                self.Lambda = 1
                
        if self.Lambda <= -0.5:
            logging.error("** Error: hyperparamter invalid range! For the Gegenbauer kernel function:  -0.5< Kernel Parameter ")
            sys.exit()
        elif -0.5 <self.Lambda <= 0.5:
            for i in range(self.order):
                temp += self.GegenbauerTerm(x, i) * self.GegenbauerTerm(y, i)
            for t in temp:
                result += result * t       
        elif self.Lambda >= 0.5:
            for i in range(self.order):
                temp += self.GegenbauerTerm(x, i) * self.GegenbauerTerm(y, i) * self.GegenbauerWeight(x, y) * self.GegenbauerScale(i) ** 2
            for t in temp:
                result += result * t
        return result



class Jacobi(object):
    """
    Jacobi kernel class, containing all functions required to compute the Jacobi kernel function.
    For sake of simplicity and according to the most well-known resources, teh main computation process has been 
    splitted to multiple functions, the An(n),Bn(n),Cn(n) functions
     are directly used in JacobiTerm function.

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
        Use in JacobiTerm for calculating Jacobi Jacobi polynomials.
    Bn(n)
        Use in JacobiTerm for calculating Jacobi Jacobi polynomials.
    Cn(n)
        Use in JacobiTerm for calculating Jacobi Jacobi polynomials.
    JacobiTerm(n, x)
        Use in kernel to calculate the jacobi polynomials.
    weight(x, z)
        Use in kernel to calculate the Jacobi kernel weight.
    kernel(x, z)
        Jacobi kernel function
    """


    def __init__(self, psi = -0.8, omega = 0.2, order=3, noise=0.1):
        """
        Constructor for Jacobi class.

        Attributes
        ----------
        psi : float
            Hyperparameter of Jacobi kernel.
            psi should be greater than -1 (KernelParam1 > -1).
        omega : float
             Hyperparameter of Jacobi kernel.
             omega should be greater than -1 (KernelParam2 > -1).
        order : int
            the order orthogonal kernel.
        noise : float
            A noise only applicable to weight function of Jacobi kernel. recommended values: 0.1, 0.01,...
        """
        self.psi = psi
        self.omega = omega
        self.order = order
        self.noise = noise
        self.name = "Jacobi"

    def An(self, n):
        """
       Used in JacobiTerm for calculating Jacobi Jacobi polynomials.

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
        Used in JacobiTerm to calculate Jacobi polynomials.

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
        Use in JacobiTerm to calculate  Jacobi polynomilas.

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
        Used in Jacobi kernel to calculate the Jacobi polynomials for given input and order.

        Parameters
        ----------
        x: numpy array / float
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
        x: numpy array / float
        y: numpy array / float

        Return
        -------
        float
           Jacobi weight.
        """
        d = len(x)     # input's dimension 
        return (((d - np.dot(x, y)) + self.noise) ** self.psi) * (((d + np.dot(x, y)) + self.noise) ** self.omega)

    def kernel(self, x, y):
        """
        Calculate the Jacobi kenrel function for given inputs.

        Parameters
        ----------
        x: numpy array
          A point in dataset.
        y: numpy array
          A point in dataset.
        Return
        -------
        float
            Calculated kernel for x , y.
        """
        if (self.order <0) :
            logging.error("order must be equal or greater than 0")
            sys.exit()
        
        if(self.psi is None) :
            self.psi = -0.8
            
        if(self.omega is None) :
            self.omega = 0.2
            
        if(self.psi<= -1) :
            logging.error("** Error: hyperparamter invalid range! For the Jacobi kernel function:  -1< Kernel Parameter1 ")
            sys.exit()
        if(self.omega<=-1) :
            logging.error("** Error: hyperparamter invalid range! For the Jacobi kernel function:  -1< Kernel Parameter2 ")
            sys.exit()
            
        _sum = 0
        result = 1

        try:
            d = len(x)    # Input's dimension
        except:
            sys.exit("input data kernel dimension has to be > 1!")
        for i in range(self.order + 1):
            _sum += self.JacobiTerm(i, x) * np.transpose(self.JacobiTerm(i, y))
        for i in _sum * self.Weight(x, y):
            result = result * i

        return result
    
    
class RBF(object) :
    """
    Radial basis function kernel class
    Attributes
    ----------
    gamma : float
        Hyperparameter of rbf kernel.gamma should be greater than zero.
    """

    def __init__(self, gamma = 0.001):
        """
        Constructor for rbf class.
        Attributes
        ----------
        gamma : float
        Hyperparameter of rbf kernel.
        """
        self.gamma = gamma

    def kernel(self, x, y):
        """
        Calculate the rbf kernel function.
        Parameters
        ----------
        x: numpy array
          A point in dataset.
        y: numpy array
          A point in dataset.
        Return
        -------
        float
            Calculated kernel for x , y.
        """
        if (self.gamma is None) :
                self.gamma = 0.001
        
        if (self.gamma <=0 ) :
            logging.error("gamma must be equal or greater than 0")
            sys.exit()
                
        x_norm = np.sum(np.power(x, 2))  # calculate norm of x
        y_norm = np.sum(np.power(y, 2))  # calculate norm of y
        result = np.exp(-self.gamma * (x_norm + y_norm - 2 * np.dot(y, x.T)))  # calculate kernel result
        return result
