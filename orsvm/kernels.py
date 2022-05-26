"""
Kernel functions

"""

import sys
import math
import numpy as np

class Chebyshev(object):
    def __init__(self, order,form):
        self.order = order
        self.name = "Chebyshev"
        self.form = form
        
        """
        This is Chebyshev class contains two functions:

        1- G_Tn used in Gegenbauer kernel:
            Vectorized method of calculating chebsyhev polynomials terms.
            Calculation of Explicit form is also considered, where 'f=e' represents the use of Explicit equation: 'np.cos(n * np.arccos(x))' and "f=r" represents recursive form

        2- kernel fucntion to calculate the cheybshev kenrel fucntion.

        """


    #Pairewise/vectorial implemantation of Chebyshev Kernel aka Generalized Chebyshev Kernel
    def G_Tn(self, x, n):
        if self.form == "e":
            """
                Explicit form
            """
            return np.cos(n * np.arccos(x))

        elif self.form == "r":
            """
                Recursive form
            """
            if n == 0:
                return 1
            elif n == 1:
                return x
            elif n >= 2:
                return 2 * x * np.transpose(self.G_Tn(x, n - 1)) - self.G_Tn(x, n - 2)
                


    def kernel(self, x, y):
        d = len(x)
        chebyshev_result = 0
        for i in range(self.order):
            chebyshev_result += (np.inner(self.G_Tn(x, i),self.G_Tn(y, i)))
        return chebyshev_result / (np.sqrt((d - np.inner(x, y))) + 0.002) # +0.002 according to Meng Tian et.al

class Legendre(object):
    def __init__(self, order=1):
        self.order = order
        self.name = "Legendre"
        
    """
    Legandre class contain two functions:
    
    1- Legendre_term used in Legandre kernel to calculate legendre polynomials
    2- kernel function 
    
    """


    def Legendre_term(self, x, n):
        try:
            if n > 1:
                return ((2 * n - 1) * (x * self.Legendre_term(x, n - 1)) - (n - 1) * self.Legendre_term(x, n - 2)) / n
            elif n == 1:
                return x
            elif n == 0:
                return 1
            return 0
        except:
            sys.exit("order must be equal or grater then 0")

    def kernel(self, x, z):
        result = 1
        temp = 0
        for i in range(self.order):
            temp += self.Legendre_term(x, i) * self.Legendre_term(z, i)
        for t in temp:
            result += result * t
        
        return result
    
    
class Chebyshev(object):
    def __init__(self, order):
        self.order = order
        self.name = "Chebyshev"
        
        """
        This is Chebyshev class contains two functions:

        1- G_Tn used in Gegenbauer kernel:
            Vectorized method of calculating chebsyhev polynomials terms.
            Calculation of Explicit form is also considered, where 'f=e' represents the use of Explicit equation: 'np.cos(n * np.arccos(x))' and "f=r" represents recursive form

        2- kernel fucntion to calculate the cheybshev kenrel fucntion.

        """


    #Pairewise/vectorial implemantation of Chebyshev Kernel aka Generalized Chebyshev Kernel
    def G_Tn(self, x, n,f='r'):
        if f == 'e':
            """
                Explicit form
            """
            return np.cos(n * np.arccos(x))

        elif f == 'r':
            """
                Recursive form
            """
            if n == 0:
                return 1
            elif n == 1:
                return x
            elif n >= 2:
                return 2 * x * np.transpose(self.G_Tn(x, n - 1)) - self.G_Tn(x, n - 2)
                


    def kernel(self, x, y,form='r'):
        d = len(x)
        chebyshev_result = 0
        for i in range(self.order):
            chebyshev_result += (np.inner(self.G_Tn(x, i),self.G_Tn(y, i)))
        return chebyshev_result / (np.sqrt((d - np.inner(x, y))) + 0.002) # +0.002 according to Meng Tian et.al
        
class Gegenbauer(object):

    def __init__(self, order=1, Lambda=1):
        self.order = order
        self.Lambda = Lambda
        self.name = "Gegenbauer"

    """
    Gegenbauer class contains four functions
    
    1- Gegenbauer_term used in Gegenbauer kernel to calculate the gegenbauer polynomials
    2- Gegenbauer_weight used in Gegenbauer kernel to calculate gegenbauer relevant weight 
    3- Gegenbauer_scale used in Gegenbauer kernel 
    4- Gegenbauer_kernel
    """

    def Gegenbauer_term(self, x,n):
        try:
            if n == 0:
                return 1
            elif n == 1:
                return 2 * self.Lambda * x
            elif n > 1:
                return ((2 * (n - 1 + self.Lambda) / n * x * self.Gegenbauer_term(x, n-1)) - (((n + (2 * self.Lambda) - 2) / n) * self.Gegenbauer_term(x, n - 2)))
            return 0
        except:
            sys.exit("order must be equal or grater then 0")

    def Gegenbauer_weight(self, x, z):
        return ((1 - x ** 2) * (1 - z ** 2)) ** (self.Lambda - 0.5) 

    def Gegenbauer_scale(self, n):
        return (math.sqrt(n + 1) * abs(self.Gegenbauer_term(1, n))) ** -1

    def kernel(self, x, z):
        result = 1
        temp = 0
        for i in range(self.order):
            temp += self.Gegenbauer_term(x, i) * self.Gegenbauer_term(z, i) * self.Gegenbauer_weight(x, z) * self.Gegenbauer_scale(i)**2
        for t in temp:
            result += result * t
        return result


class Jacobi(object):
    """
        Jacobi Class contains 6 functions:
        1- An
        2- Bn
        3- Cn
            3 parts of Jacobi polynomials , used in Jacobi_term, to calculate Jacobi polynomilas.
        4- Jacobi_term, used in kernel to calculate the jacobi polynimials
        5- weight used in kernel to calculate the Jacobi kernel weight
        6- kenrel, the Jacobi kernel function


    """

    def __init__(self, psi,omega,order=3,noise=0.1):
        self.psi = psi
        self.omega = omega
        self.order = order
        self.noise = noise
        self.name = "Jacobi"

        
        
        
    def An(self,n):
        return ((2*n + self.psi + self.omega + 1) * (2*n + self.psi + self.omega + 2))/(2*(n + 1) * (n + self.psi + self.omega + 1))
    def Bn(self,n):
        return (((self.omega**2)-(self.psi**2))*(2*n + self.psi + self.omega +1 ))/(2*(n+1)*(n+self.psi+self.omega+1)*((2*n)+self.psi+self.omega))
    def Cn(self,n):
        return ((n+self.psi)*(n+self.omega)*(2*n+self.psi+self.omega+2))/((n+1)*(n+self.psi+self.omega+1)*(2*n+self.psi+self.omega))

    def Jacobi_term(self,n,x):
        if n==0:
            return 1
        if n==1:
            return 0.5*np.dot((self.psi+self.omega+2),x)+ 0.5*(self.psi-self.omega)
        else:
            return ((np.dot(self.An(n-1),x)+self.Bn(n-1))*self.Jacobi_term(n-1,x)) - (self.Cn(n-1) * self.Jacobi_term(n-2,x))

    def weight(self,x,z):
        d=len(x)
        return (((d-np.dot(x,z))+self.noise)**self.psi) * (((d+np.dot(x,z))+self.noise)**self.omega)
        
    def kernel(self,x,z):
        _sum=0
        result=1
        
        try:
            d=len(x)
        except:
            sys.exit("input data kernel dimension has to be > 1!")        
        for i in range(self.order+1):
             _sum += self.Jacobi_term(i,x)*np.transpose(self.Jacobi_term(i,z))
        for i in _sum * self.weight(x,z):
            result= result * i
        
        return result

