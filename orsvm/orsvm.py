import cvxopt
import logging
import sys
import math
import json
import numpy as np
import cvxopt.solvers
from decimal import Decimal
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from .kernels import Jacobi, Legendre, Gegenbauer, Chebyshev,RBF

cvxopt.solvers.options['show_progress'] = False
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

def Transform(x, T=1):
    """
    Transforms/normalizes data into the selected kernel space.
    
    Parameters
    ----------
    x : Array
        The set to be normalized or transformed to a new space.
    T : float, default is 1
        Fractional Order, if 0 < T < 1, then transformation applies on input data-set.
           If T=1, then normalization applies on to input data-set.

    Returns
    -------
    Numpy Array         Normalized/Transformed data-set.

    """
    min = np.min(x)  # min of x
    max = np.max(x)  # max of x
    new_x = [2 * (((p - min) / (max - min)) ** (T)) - 1 for p in x]  # transform/normalize x and make new list
    return np.asarray(new_x)  # retrun transformed/normilized x

def SupportMultipliers(array, n, order='asc'):
    """
    Function which determines the support vectors among Lagrange multipliers.

    Parameters
    ----------
    array : numpy array
        The Lagrange Multipliers.
    n : char , int , scientific notation
        If it's a character == "A", then the average method is chosen.
        If it's a positive integer, then its the number of support vectors that should be chosen from Lagrange Multipliers.
        If it's greater than the number of Lagrange Multipliers then all of them will be chosen.
        If it's a number expressed using scientific notation (ex: 1e-5), that's the minimum threshold,
        All the support vectors are SV >=  1e-5

    Returns
    -------
    numpy array
        Support Multipliers.
    """

    def _IsScientific(n):
        """
           Function which determines whether the argument n is a number expressed using scientific notation (ex: 1e-5) or not.

           Parameters
           ----------
           n : char , int , scientific notation
                SVM determiner.

           Returns
           -------
           bool
                number expressed is scientific notation or not.
           """

        if 'e' in str(n).lower():
            return True
        else:
            return False

    def _FormatE(nn):
        a = '%E' % nn
        return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

    def _ScientificNotationParser(x):

        if "-" in x:
            txt_mod = (x.split("-", 2)[-1])
            if "+" in txt_mod:
                return int(x.split("+", 2)[-1])
            else:
                return -(int(txt_mod))
        elif "+" in x:
            txt_mod = x.split("+", 2)[-1]
            return int(txt_mod)

    # select directly first n elements of Lagrange Multipliers whether in asc order or desc order
    if isinstance(n, int):
        if (n>len(array)) :
            logging.info('** Strictly all of (%s) support vectors are selected !',len(array))
        else :    
            logging.info('** Strictly %s first elements of support vectors are selected!',n)
        
        if order not in ('asc', 'desc'):
            logging.error("** Error: order should be 'asc' or 'desc'")
            sys.exit()
        elif order == 'asc':
            options = np.sort(array)[:n]  # Support Multipliers in asc order
            support_multipliers = np.isin(array,options)
        elif order == 'desc':
            options = np.sort(array)[::-1][:n]  # Support Multipliers in desc order
            support_multipliers = np.isin(array,options)

    # if n is a scientific number, select Support vectors that are greater than n
    elif _IsScientific(n):
        logging.info("** Scientific number %s is received as threshold for support vectors",n)
        support_multipliers = array > float(n)

    # if n is a character 'a', the average of Support vectors is calculated. Select Support vectors that are greater than 10^average

    elif str(n).lower() == 'a':
        logging.info("** Average method for support vector determination selected!")
        sum = 0
        for i in array:
            c = _ScientificNotationParser(str(_FormatE(Decimal(i))))
            sum += c
        sv_treshold = int(sum / len(array))

        logging.info("** support vector threshold: 10^%s", sv_treshold)
        support_multipliers = array > 10 ** +sv_treshold
    else:
        logging.error('support vector determiner is not valid')
        sys.exit()
    return support_multipliers


class SVM(object):
    """
    SVM class uses to represent SVM which is used by the Model class.

    Attributes
    ----------
    kernel : str
        Kernel name, can be one of: Legendre, Jacobi, Gegenbauer, Chebyshev , rbf.
    order : int
        Orthogonal kernel's order.
    T : float
        The Transition order. 0<T<=1. default is 1, which means transition to normal space (normalization) instead of transformation.
    KernelParam1 : float
        First hyperparameter of the kernels for Jacobi ,Gegenbauer and rbf.
        Jacobi: KernelParam1 refers to psi, first hyperparameters of jacobi kernel. psi should be greater than -1.
        Gegenbauer: KernelParam1 refers to lambda, the hyperparameter of Gegenbauer kernel. lambda should be greater than -0.5.
        rbf : KernelParam1 refers to gamma, the hyperparameter of rbf kernel. gamma should be greater than zero.
    KernelParam2 : float
        Second hyperparameter of the kernels for Jacobi.
        Jacobi: KernelParam2 refers to the omega, second hyperparameter of the jacobi kernel. omega should be greater than -1.
    SupportVectorDeterminer : char , int , scientific number , optional
         Support vector determiner. 3 approaches are possible to choose the support vectors
         among Lagrange Multipliers:
             1-  SupportVectorDeterminer = int  :  User chooses how many Lagrange multipliers, consider being
                 support vectors.
             2-  SupportVectorDeterminer = 1e-3 :  User may choose a minimum threshold to choose support vectors from
                 Lagrange multipliers. In this case, the passed argument must be in scientific notation
                 for example 1e-3, chooses all Lagrange multipliers that are greater than '1e-3'(0.001)
             3-  SupportVectorDeterminer = 'a'  :  (default) setting the svd = 'a' and ORSVM will compute the average
                 of scientific notation of the scale that Lagrange Multipliers are in, and sets as the
                 criteria to select the ones that are greater than average. Choosing the "average" may
                 cause most of Lagrange Multipliers to be selected as Support Vectors!
                 And this may lead to poor generalization of the fitting result! but the benefit of this
                 one is whenever the user may not know how to choose the threshold, in this case, choosing
                 the wrong threshold may outcomes 0 Support Vectors, therefore setting svd = 'a' for the
                 first, fit gives an intuition to choose the best value.
    form : CHAR
          Possible values : 'r' or 'e'
          which chooses how the Chebyshev kernel is to be used whether explicit or recursive.
          Only is applicable to the Chebyshev kernel, for other kernels if passed, will be ignored.
    C : int , optional
        The regularization parameter of SVM algorithm.
        The default is None. Possible values: 10, 100, 1000 ...
    noise: float
        Noise is only applicable to the weight function of the Jacobi kernel. recommended values can be 0.1, 0.01,...

    Methods
    -------
    Fit(x_train, y_train, log=True)
         The main fitting function, to invoke the cvxopt and solve the convex SVM equation.

    Predict(x, b, kernel)
        Prediction function, to predict using kernel instance and bias.
    """

    def __init__(self, kernel,order,T,KernelParam1,KernelParam2,svd,form,C,noise):
        """
        Constructor of SVM class.

        Parameters
        ----------
        kernel : str
            Kernel name, can be one of: Legendre, Jacobi, Gegenbauer, Chebyshev , rbf.
        order : int
            Orthogonal kernel's order.
        T : float
            The Transition order. 0<T<=1. default is 1, which means transition to normal space (normalization) instead of transformation.
        KernelParam1 : float
            First hyperparameter of the kernels for Jacobi and Gegenbauer and rbf.
            Jacobi: KernelParam1 refers to psi, first hyperparameters of jacobi kernel. psi should be greater than -1.
            Gegenbauer: KernelParam1 refers to lambda, the hyperparameter of Gegenbauer kernel. lambda should be greater than -0.5.
            rbf : KernelParam1 refers to gamma, the hyperparameter of rbf kernel. gamma should be greater than zero.
        KernelParam2 : float
            Second hyperparameter of the kernels for Jacobi.
            Jacobi: KernelParam2 refers to the omega, second hyperparameter of the jacobi kernel. omega should be greater than -1.
        svd : char , int , scientific number , optional
             Support vector determiner. 3 approaches are possible to choose the support vectors
             among Lagrange Multipliers:
                 1-  svd = int  :  User chooses how many Lagrange multipliers, consider to be
                     support vectors.
                 2-  svd = 1e-3 :  User may choose a minimum threshold to choose support vectors from
                     Lagrange multipliers. In this case, the passed argument must be in scientific notation
                     for example 1e-3, chooses all Lagrange multipliers that are greater than '1e-3'(0.001)
                 3-  svd = 'a'  :  (default) setting the svd = 'a'  and ORSVM will compute the average
                     of scientific notation of the scale that Lagrange Multipliers are in, and sets as the
                     criteria to select the ones that are greater than average. Choosing the "average" may
                     cause most of Lagrange Multipliers to be selected as Support Vectors!
                     And this may lead to poor generalization of the fitting result! but the benefit of this
                     one is whenever the user may not know how to choose the threshold, in this case, choosing
                     the wrong threshold may outcomes 0 Support Vectors, therefore setting svd = 'a' for the
                     first fit gives an intuition to choose the best value.
        form : CHAR
            Possible values : 'r' or 'e'
              which chooses how the Chebyshev kernel is to be used whether explicit or recursive.
              Only is applicable to the Chebyshev kernel, for other kernels if passed, will be ignored.
        C : int
            Is the regularization parameter of the SVM algorithm
            The default is None. Possible values: 10, 100, 1000 ...
        noise: float
            Noise is only applicable to the weight function of the Jacobi kernel. recommended values can be 0.1, 0.01,...
            """
        self.kernel = str(kernel).lower()
        self.order = order
        self.T = T
        self.KernelParam1 = KernelParam1
        self.KernelParam2 = KernelParam2
        self.SupportVectorDeterminer = svd
        self.form = form
        self.C = C
        self.noise = noise

        if self.C is not None:
            self.C = float(self.C)

    def Fit(self, x_train, y_train, log=False):
        """
        The main fitting function, is to invoke the cvxopt and solve the convex SVM equation.
            
        Parameters
        ----------
        x_train: numpy array
            training set without labels
        y_train: numpy array
            training set only the labels
        log: bool
            If True print fitting information

        Returns
        -------
        numpy array
            weight vector.
        numpy array
            support vectors.
        float
            bias.
        object
            kernel instance.
        """

        # Making Gram Matrix
        n_samples, n_features = x_train.shape
        K = np.zeros((n_samples, n_samples))  # initialize kernel matrix with the shape of (n_samples, n_samples)

        if self.kernel == "legendre":
            kernel_ins = Legendre(self.order)  # making kernel instance
            # filling kernel matrix by invoking kernel method of kernel instance
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = kernel_ins.kernel(x_train[i], x_train[j])

        elif self.kernel == "jacobi":
            kernel_ins = Jacobi(self.KernelParam1, self.KernelParam2, self.order, self.noise)  # making kernel instance
            # filling kernel matrix by invoking kernel method of kernel instance
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = kernel_ins.kernel(x_train[i], x_train[j])

        elif self.kernel == "gegenbauer":
            kernel_ins = Gegenbauer(self.order, self.KernelParam1)  # making kernel instance
            # filling kernel matrix by invoking kernel method of kernel instance
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = kernel_ins.kernel(x_train[i], x_train[j])

        elif self.kernel == "chebyshev":
            kernel_ins = Chebyshev(self.order, self.form)  # making kernel instance
            # filling kernel matrix by invoking kernel method of kernel instance
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = kernel_ins.kernel(x_train[i], x_train[j])
                    
        elif self.kernel == "rbf":
            kernel_ins = RBF(self.KernelParam1)  # making kernel instance
            # filling kernel matrix by invoking kernel method of kernel instance
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = kernel_ins.kernel(x_train[i], x_train[j])

        y = y_train.astype(float)  # change y_train type to float
        """
        making cvxopt matrix for solving convex SVM equation
        """
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        # If it's Hard Margin
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:  # if it's  Soft Margin, the regularization parameter "C" comes into play
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # Solve QP problem using cvxopt quadratic programming solver.
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Get the Lagrange multipliers (a)
        lagrange_multipliers = np.ravel(solution['x'])

        """
        Support vectors have non-zero Lagrange multipliers
        3 Ways to choose support vectors from Lagrange Multipliers(LMs)
        - First receive the minimum threshold from user and choose all Lagrange multipliers that are greater than the minimum threshold
        - Second, finding the average and choosing half of the Lagrange multipliers
        - Third, user chooses a number "n" of them strictly, then Lagrange multipliers will be sorted
          in asc/desc order and the first "n" number of them will be chosen as Support Vectors
        """

        support_m_indices = SupportMultipliers(lagrange_multipliers, self.SupportVectorDeterminer)  # Compute Support Vectors
        self.support_multipliers = lagrange_multipliers[support_m_indices]  # select support_multipliers from Lagrange multipliers corresponding to support multipliers indices
        self.support_vectors = x_train[ support_m_indices]  # Support Vectors are points from input data corresponding to support multipliers indices
        self.support_vector_labels = y[support_m_indices]  # label of Support Vectors
        self._weights = self.support_multipliers  # set support multipliers to weights
        self.solution_status = solution['status']
        
        
        if np.all(np.linalg.eigvals(K) >= 0) == True:
            logging.info("Kenrel matrix is convex")
            logging.info("** solution status: %s", solution['status'])
        elif np.all(np.linalg.eigvals(K) >= 0) == False:
            logging.warning("A convex kernel matrix is not formed!")
            logging.warning("** solution status: %s", solution['status'])

        """
        If log is True some information about support multipliers and the solution of the convex equation will be logged
        """
        if log == True:
            logging.info("** Lagrange multipliers: %s", self.support_multipliers)
            logging.info("** Max is: %s Min is: %s", np.max(self.support_multipliers), np.min(self.support_multipliers))
            logging.info("** solution status: %s", solution['status'])
            logging.info("** K condition value: %s", np.linalg.cond(K))
            New_matrix = np.hstack((P, np.transpose(A)))
            New_matrix = np.hstack((np.transpose(G), New_matrix))
            logging.info("** [P, A_transpose, G_transpose] condition value: %s", np.linalg.cond(New_matrix))
            logging.info("** Rank A is: %s Rank [P, A_transpose, G_transpose] is: %s", np.linalg.matrix_rank(A),np.linalg.matrix_rank(New_matrix))
            logging.info("kernel matrix is positive semidefinite : %s" ,np.all(np.linalg.eigvals(K) >= 0))
            logging.info("** %s support vectors are selected out of %s points", len(self.support_vectors), n_samples)

        """
        A prediction on support vectors with bias = 0 and computing the error gives the bias
        """
        ind = np.arange(len(lagrange_multipliers))[support_m_indices]
        _bias = 0
        for n in range(len(self.support_multipliers)):
            _bias += self.support_vector_labels[n]
            _bias -= np.sum(self.support_multipliers * self.support_vector_labels * K[ind[n], support_m_indices])

        _bias /= len(self.support_multipliers)
        return self._weights, self.support_vectors, _bias, kernel_ins

    def Predict(self, x, b, kernel):
        """
        Function which predicts the corresponding labels according to input,bias,and kernel.

        Parameters
        ----------
        x : numpy array
            the input array to predict the corresponding labels according to fitted model.
        b : Float
            The bias to be used for prediction.
        kernel : Object
            The kernel class.

        Returns
        -------
        Array
            The array of predicted labels.
        """
        if self.kernel == 'linear':
            return np.dot(x, self._weights) + b
        else:
            y_predict = np.zeros(len(x))  # initialize y_pridct with a size equal to the number of x samples

            """
            making predictions with  support_multipliers, support_vector_labels, support_vectors and kernel function
            """
            for i in range(len(x)):
                s = 0
                for a, sv_y, sv in zip(self.support_multipliers, self.support_vector_labels, self.support_vectors):
                    s += a * sv_y * kernel.kernel(x[i], sv)
                y_predict[i] = s

            # adding bias to y_predict and applying sign function to it for predicting labels
            return np.sign(y_predict + b)

class Model(object):
    """
    The main class that should be initiated first. This class initiates the requested model and parameters.
    Creating an object of this class by passing relevant parameters outcomes an object ready to be use in fitting procedure of a data-set.

    Attributes
    ----------
    kernel : str , default is Chebyshev
            Kernel name, can be one of: Legendre, Jacobi, Gegenbauer, Chebyshev , rbf.
    order: int
        Orthogonal kernel's order.
    T: float
        the Transition order. 0<T<=1. default is 1, which means transition to normal space (normalization) instead of transformation.
    KernelParam1 : float, optional
        Default is None.
        First hyperparameter of the kernels for Jacobi, Gegenbauer , rbf.
        Jacobi: KernelParam1 refers to psi, first hyper parameters of jacobi kernel. psi should be greater than -1.
        Gegenbauer: KernelParam1 refers to lambda, the hyperparameter of Gegenbauer kernel. lambda should be greater than -0.5.
        rbf : KernelParam1 refers to gamma, the hyperparameter of rbf kernel. gamma should be greater than zero.
    KernelParam2: float, optional
        Default is None.
        Second hyperparameter of the kernels for Jacobi.
        Jacobi: KernelParam2 refers to the omega, second hyperparameter of the jacobi kernel. omega should be greater than -1.
    SupportVectorDeterminer : char , int , scientific number , optional , The default is 'a'.
         Support vector determiner. 3 approaches are possible to choosing the support vectors
         among Lagrange Multipliers:
         1-  SupportVectorDeterminer = int  :  User chooses how many Lagrange multipliers, consider to be
        support vectors.
        2-  SupportVectorDeterminer = 1e-3 :  User may choose a minimum threshold to choose support vectors from
        Lagrange multipliers. In this case, the passed argument must be in scientific notation
        for example 1e-3, chooses all Lagrange multipliers that are greater than '1e-3'(0.001)
        3-  SupportVectorDeterminer = 'a'  :  (default) setting the svd = 'a' and ORSVM will compute the average
        of scientific notation of the scale that Lagrange Multipliers are in, and sets as the
        criteria to select the ones that are greater than average. Choosing the "average" may
        cause most of Lagrange Multipliers to be selected as Support Vectors!
        And this may lead to poor generalization of the fitting result! but the benefit of this
        one is whenever the user may not know how to choose the threshold, in this case, choosing
        the wrong threshold may outcomes 0 Support Vectors, therefore setting sv_determiner = 'a' for the
        first fit gives an intuition to choose the best value.
    form : CHAR, optional , The default is 'r'
         Possible values : 'r' or 'e'
                which chooses how the Chebyshev kernel to be used whether explicit or recursive.
                Only is applicable to Chebyshev kernel, for other kernels if passed, will be ignored.
    C : int, optional
        Is the regularization parameter of SVM algorithm
        The default is None. Possible values: 10, 100, 1000 ...
    noise: float
        Noise is only applicable to the weight function of the Jacobi kernel. recommended values can be 0.1, 0.01,...

    Methods
    -------
        ModelFit(x_train,y_train)
            The fit function of Model class. The dataset has to be provided into two sets as train and test sets.
        ModelPredict(x_test, y_test,bias, k_ins)
            Predict function of Model class. Using the test-set returns the accuracy score of the model.
    """
    def __init__(self,kernel="Chebyshev",order=2,T=1,KernelParam1=None,KernelParam2=None,svd='a',form='r',c=None,noise=0.1,orsvmModel=None):
        """
        Constructor of Model class

        Parameters
        ----------
        kernel : str , default is Chebyshev
            Kernel name, can be one of: Legendre, Jacobi, Gegenbauer, Chebyshev , rbf.
        order: int
            Orthogonal kernel's order.
        T: float
            the Transition order. 0<T<=1. default is 1, which means transition to normal space (normalization) instead of transformation.
        KernelParam1 : float, optional
            Default is None.
            First hyperparameter of the kernels for Jacobi and Gegenbauer , rbf.
            Jacobi: KernelParam1 refers to psi, first hyper parameters of jacobi kernel. psi should be greater than -1.
            Gegenbauer: KernelParam1 refers to lambda, the hyperparameter of Gegenbauer kernel. lambda should be greater than -0.5.
             rbf : KernelParam1 refers to gamma, the hyperparameter of rbf kernel. gamma should be greater than zero.
        KernelParam2: float, optional
            Default is None.
            Second hyperparameter of the kernels for Jacobi.
            Jacobi: KernelParam2 refers to the omega, second hyperparameter of the jacobi kernel. omega should be greater than -1.
        svd : char , int , scientific number , optional , The default is 'a'.
             Support vector determiner. 3 approaches are possible to choosing the support vectors
             among Lagrange Multipliers:
             1-  svd = int  :  User chooses how many Lagrange multipliers, consider to be
            support vectors.
            2-  svd = 1e-3 :  User may choose a minimum threshold to choose support vectors from
            Lagrange multipliers. In this case, the passed argument must be in scientific notation
            for example 1e-3, chooses all Lagrange multipliers that are greater than '1e-3'(0.001)
            3-  svd = 'a'  :  (default) setting the svd = 'a' and ORSVM will compute the average
            of scientific notation of the scale that Lagrange Multipliers are in, and sets as the
            criteria to select the ones that are greater than average. Choosing the "average" may
            cause most of Lagrange Multipliers to be selected as Support Vectors!
            And this may lead to poor generalization of the fitting result! but the benefit of this
            one is whenever the user may not know how to choose the threshold, in this case, choosing
            the wrong threshold may outcomes 0 Support Vectors, therefore setting sv_determiner = 'a' for the
            first fit gives an intuition to choose the best value.
        form : CHAR, optional , The default is 'r'
             Possible values : 'r' or 'e'
                    which chooses how the Chebyshev kernel to be used whether explicit or recursive.
                    Only is applicable to Chebyshev kernel, for other kernels if passed, will be ignored.
        C : int, optional
            Is the regularization parameter of SVM algorithm
            The default is None. Possible values: 10, 100, 1000 ...
        noise: float
            Noise is only applicable to the weight function of the Jacobi kernel. recommended values can be 0.1, 0.01,...
        """
        self.Kernel = str(kernel).lower()
        self.Order = order
        self.T = T
        self.KernelParam1 = KernelParam1
        self.KernelParam2 = KernelParam2
        self.SupportVectorDeterminer = svd
        self.Form = form
        self.C = c
        self.Noise = noise
        self.OrsvmModel = orsvmModel

    def ModelFit(self, x_train, y_train):
        """
            This is the fit function of the Model class. The data-set has to be provided into two sets as train and test sets.

            Parameters
            ----------
            x_train : numpy array
                 training data
            y_train : numpy array
                training labels

            Returns
            -------
            numpy array
                Weight vector.
            numpy array
                Support vectors.
            float
                Bias.
            object
                Kernel instance.
        """
        """
        check whether kernel name is valid or not
        """
        if self.Kernel != "legendre" and self.Kernel != "jacobi" and self.Kernel != "gegenbauer" and self.Kernel != "chebyshev" and self.Kernel != "rbf": 
            logging.error("Kernel name is not valid!")
            sys.exit()
            
        if (x_train.shape[0]!=y_train.shape[0]) : # check whether x_train.shape is compatible with y_train.shape or not
            logging.error("x_train shape is not compatible with y_train shape!")
            sys.exit()
            
        elif(x_train.shape[0] == 0 or y_train.shape[0] == 0 ) : # check whether n_sample = 0 or not
            logging.error(" Model can not fit with n_sample = 0 ")
            sys.exit()
            
        if (self.T<=0 or self.T>1) : # check range of T
            logging.error(" T is out of range. T range is : 0 < T <=1 ")
            sys.exit()
            
        if (self.C < 0) :
            logging.error(" C is out of range. C must be greater than 0 ")
            sys.exit()
        
        y_unique_values = np.unique(y_train)  # get unique labels in y_train

        if(len(y_unique_values) == 2)  :  # if it is binary classification, map zero labels to -1
            y_train [y_train == 0] = -1
            
        #TODO
        elif(len(y_unique_values) > 2) :
            logging.info("Multiclassification")
            sys.exit()
            
        """
        Some information will be logged about Model 
        """    
        # print("*" * 10, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "*" * 10)
        logging.info("** ORSVM kernel: %s", self.Kernel)
        logging.info("** Order: %s", self.Order)

        if self.T < 1:
            logging.info("** Fractional mode, transition : %s", self.T)

        else:
            logging.info("** Normal Mode, Transition: %s", self.T)

        x_train = Transform(x_train, self.T)  # transform x_train based on T parameter

        # Create an object of SVM class with proper parameters with orthogonal kernel
        self.OrsvmModel = SVM(self.Kernel, self.Order, self.T, self.KernelParam1, self.KernelParam2,self.SupportVectorDeterminer, self.Form, self.C, self.Noise)

        # Capture the Weights, Support Vectors, and the Kernel itself for future use
        WeightVector, SupportVectors, Bias, kernelInstance = self.OrsvmModel.Fit(x_train, y_train)

        return WeightVector, SupportVectors, Bias, kernelInstance  # Fit the data and return the result

    def ModelPredict(self, x_test, y_test, bias, k_ins):
        """
            Predict function of Model class. Using the test-set returns the accuracy score of the model.

            Parameters
            ----------
            x_test : numpy array
                Test set.
            y_test : numpy array
                Test set labels.
            bias : float
                The bias of hyperplane's equation.
            k_ins : obj
                Fitted kernel model.

            Returns
            -------
            float
               Accuracy_score.
        """
        if (x_test.shape[0]!=y_test.shape[0]) : # check whether x_train.shape is compatible with y_train.shape or not
            logging.error("x_test shape is not compatible with y_test shape!")
            sys.exit()
            
        elif(x_test.shape[0] == 0 or y_test.shape[0] == 0) : # check whether n_sample = 0 or not
            logging.error(" Model can not predict with n_sample = 0 ")
            sys.exit()
        
        y_unique_values = np.unique(y_test)  # get unique labels in y_test

        if (len(y_unique_values) == 2):  # if it is binary classification, map zero labels to -1
            y_test[y_test == 0] = -1
        
        #TODO
        elif (len(y_unique_values) > 2):
            logging.info("Multiclassification")
            sys.exit()
            
        x_test = Transform(x_test, self.T)  # Transform the test-set into new space
        y_predict = self.OrsvmModel.Predict(x_test, bias, k_ins)  # find the accuracy

        """
        Some information will be logged about model performance 
        """
        logging.info("** Accuracy score: %s", accuracy_score(y_test, y_predict))  # Accuracy score
        logging.info("** Classification Report: \n %s ",classification_report(y_test, y_predict))  # Classification Report
        logging.info("** Confusion Matrix: \n %s", confusion_matrix(y_test, y_predict))  # Confusion Matrix

        return accuracy_score(y_test, y_predict)  # Return the accuracy score

    def KernelMatrix(self, x, y, transform=False):
        """ 
           Function to return only the kernel matrix (a.k.a. Hermitian matrix) without any calculation for the support vector.
           Suggested to use as a precomputed mode of the custom kernel in SVM/SVC of sklearn.

            Parameters
            ----------
            x : numpy array
            y : numpy array
            transform : bool , default is False
                if transform is True x ,y will be transformed with the Transform function

            Returns
            -------
            numpy array
                Kernel matrix for our custom kernel.
                """
        if (x.shape[0] == 0 or y.shape[0] == 0):  # check whether n_sample = 0 or not
            logging.error(" Kernel matrix can not be computed with n_sample = 0 ")
            sys.exit()
        n_samples_x, n_features_x = x.shape
        n_samples_y, n_features_y = y.shape
        K = np.zeros((n_samples_x, n_samples_y))  # initialize the kernel matrix

        transformed_x = x
        transformed_y = y

        # if transform is True, use the Transform function
        if transform:
            transformed_x = Transform(x, self.T)
            transformed_y = Transform(y, self.T)

        if self.Kernel == 'chebyshev':
            kernel_ins = Chebyshev(self.Order, self.Form)  # making kernel instance
            # filling kernel matrix by invoking kernel method of kernel instance
            for i in range(n_samples_x):
                for j in range(n_samples_y):
                    K[i, j] = kernel_ins.kernel(transformed_x[i], transformed_y[j])  
            return K

        elif self.Kernel == "gegenbauer":
            kernel_ins = Gegenbauer(self.Order, self.KernelParam1)  # making kernel instance
            # filling kernel matrix by invoking kernel method of kernel instance
            for i in range(n_samples_x):
                for j in range(n_samples_y):
                    K[i, j] = kernel_ins.kernel(transformed_x[i], transformed_y[j])
            return K

        elif self.Kernel == "jacobi":
            kernel_ins = Jacobi(self.KernelParam1, self.KernelParam2, self.Order, self.Noise)  # making kernel instance
            # filling kernel matrix by invoking kernel method of kernel instance
            for i in range(n_samples_x):
                for j in range(n_samples_y):
                    K[i, j] = kernel_ins.kernel(transformed_x[i], transformed_y[j])
            return K

        elif self.Kernel == "legendre":
            kernel_ins = Legendre(self.Order)  # making kernel instance
            # filling kernel matrix by invoking kernel method of kernel instance
            for i in range(n_samples_x):
                for j in range(n_samples_y):
                    K[i, j] = kernel_ins.kernel(transformed_x[i], transformed_y[j])
            return K
                                                 
        else: # check the validity of kernel name
            logging.error("Kernel name is not valid")
            sys.exit()

    def SaveToJason(self, path = None, Weights=None, SupportVectors=None, KernelMatrix=None, Bias=None, accuracy=None):
        """
        Function to save the fitting results into to the path. The output will be save in jason format.

        Parameters
        ----------
        path : str
            Path to save output file.
        Weights : numpy array , default is None (optional)
            Weights matrix which is returned from ModelFit function.
        SupportVectors : numpy array , default is None (optional)
            SupportVectors matrix which is returned from ModelFit function.
        KernelMatrix : numpy array , default is None (optional)
            KernelMatrix which is returned from KernelMatrix function.
        Bias : float , default is None (optional)
           Bias number which is returned from ModelFit function.
        accuracy : float
            Accuracy score which is returned from ModelPredict function.

        Returns
        ----------
        If path is None, returns the resulting json format as a dictionary. Otherwise returns None.

        """

        mode = "normal"
        if (self.T < 1):
            mode = "fractional"

        """
        Convert numpy arrays to lists as numpy arrays can not be serialized in json files
        """
        if isinstance(Weights, np.ndarray):
            Weights = Weights.tolist()

        if isinstance(SupportVectors, np.ndarray):
            SupportVectors = SupportVectors.tolist()

        if isinstance(KernelMatrix, np.ndarray):
            KernelMatrix = KernelMatrix.tolist()
        
        status = None
        support_vector_labels = None
        support_multipliers = None
        
         # set support_multipliers and support_vector_labels if they are not Nonetype
        if(self.OrsvmModel is not None) :
            status = self.OrsvmModel.solution_status
            if(self.OrsvmModel.support_multipliers is not None) :
                if(len(self.OrsvmModel.support_multipliers.tolist())!=0) :
                    support_multipliers = self.OrsvmModel.support_multipliers.tolist()
            if(self.OrsvmModel.support_vector_labels is not None) :
                if(len(self.OrsvmModel.support_vector_labels.tolist())!=0):
                    support_vector_labels = self.OrsvmModel.support_vector_labels.tolist()

        # make output dictionary
        OutputDict = {
            "kernel": self.Kernel,
            "kernelParam1" : self.KernelParam1 ,
            "kernelParam2" : self.KernelParam2,
            "order": self.Order,
            "form" : self.Form,
            "noise" : self.Noise,
            "mode": mode,
            "C" : self.C,
            "transition": self.T,
            "svd": self.SupportVectorDeterminer,
            "support multipliers": support_multipliers,
            "support vectors": SupportVectors,
            "support vector labels" : support_vector_labels ,
            "weights": Weights,
            "kernel matrix": KernelMatrix,
            "bias": Bias,
            "accuracy": accuracy,
            "status": status

        }

        # if path is None return output as a dictionary
        if path is None:
            return OutputDict

        # write output dictionary to json file
        else:

            try:
                with open(path, "w") as OutFile:
                    json.dump(OutputDict, OutFile)
                logging.info("Results saved successfully")

            except FileNotFoundError:
                logging.error("Path to save output not found!")

            return None


def LoadJason(path):
    """
    Function that imports the previously stored fitting results and parameters and returns it as a dictionary.

    Parameters
    ----------
    path : str
        Path to load json file.
    Returns
    ----------
    dict
    InputFile results.
    """
    OutputDict = {}

    try:
        with open(path, 'r') as openfile:
            # Reading from json file
            OutputDict = json.load(openfile)

        """
         convert support vectors, weights , kernel matrix , support multipliers and support vector labels to their original type (numpy array)
        """
        if isinstance(OutputDict["support vectors"], list):
            OutputDict["support vectors"] = np.array(OutputDict["support vectors"])

        if isinstance(OutputDict["weights"], list):
            OutputDict["weights"] = np.array(OutputDict["weights"])

        if isinstance(OutputDict['kernel matrix'], list):
            OutputDict['kernel matrix'] = np.array(OutputDict['kernel matrix'])
                                                  
        if isinstance(OutputDict['support multipliers'] ,list) :
            OutputDict['support multipliers'] = np.array(OutputDict['support multipliers'])

        if isinstance(OutputDict['support vector labels'] ,list) :
            OutputDict['support vector labels'] = np.array(OutputDict['support vector labels'])


    except FileNotFoundError:
        logging.error("Path to load output file not found!")
        sys.exit()

    return OutputDict



def PredictWithJson(path, X_test, y_test) :
    """
    Function that predicts labels with the previously stored fitting results and returns the accuracy score of the model.

    Parameters
    ----------
    path : str
        Path to load json file.
    X_test : numpy array
        test set
    y_test : numpy array
        test labels

    Returns
    -------
    float
       Accuracy_score.
    """
    if (X_test.shape[0] == 0 or y_test.shape[0] == 0): # check whether n_sample = 0 or not
        logging.error(" Model can not predict with n_sample = 0 ")
        sys.exit()

    """
    Load JSON file and read previously stored fitting results from the dictionary
    """

    OutputDict = LoadJason(path)
    kernel = OutputDict['kernel']
    order = OutputDict['order']
    form = OutputDict['form']
    kernelParam1 = OutputDict['kernelParam1']
    kernelParam2 = OutputDict['kernelParam2']
    T = OutputDict['transition']
    C = OutputDict['C']
    noise = OutputDict['noise']
    bias = OutputDict['bias']
    SupportVectorDeterminer = OutputDict["svd"]

    """
    Create an object of SVM class with proper parameters with orthogonal kernel
    """

    orsvmModel = SVM(kernel, order, T, kernelParam1, kernelParam2, SupportVectorDeterminer, form, C, noise)
    """
    Check whether support multipliers , support vectors and support vector labels are Nonetype or not
    """
    if(OutputDict["support multipliers"] is None) :
        logging.error('Model can not predict with support multipliers of type None ')
        sys.exit()
    if(OutputDict['support vectors'] is None) :
        logging.error('Model can not predict with support vectors of type None ')
        sys.exit()
    if(OutputDict['support vector labels'] is None) :
        logging.error('Model can not predict with support vector labels of type None ')
        sys.exit()                                              
                                                  
                                                  
    orsvmModel.support_multipliers = OutputDict["support multipliers"]
    orsvmModel.support_vectors = OutputDict['support vectors']
    orsvmModel.support_vector_labels = OutputDict['support vector labels']

    obj = Model(kernel=kernel, order=order, KernelParam1=kernelParam1, KernelParam2=kernelParam2, T=T, noise=noise ,orsvmModel=orsvmModel) # create an object of Model class with proper parameters


    if kernel == 'chebyshev':
        kernelInstance = Chebyshev(order, form)  # making kernel instance
        accuracy = obj.ModelPredict(X_test, y_test, bias, kernelInstance) # call ModelPredict function
        return accuracy


    elif kernel == "gegenbauer":
        kernelInstance = Gegenbauer(order, kernelParam1)  # making kernel instance
        accuracy = obj.ModelPredict(X_test, y_test, bias, kernelInstance) # call ModelPredict function
        return accuracy

    elif kernel == "jacobi":
        kernelInstance = Jacobi(kernelParam1, kernelParam2, order, noise)  # making kernel instance
        accuracy = obj.ModelPredict(X_test, y_test, bias, kernelInstance) # call ModelPredict function
        return accuracy


    elif kernel == "legendre":
        kernelInstance = Legendre(order)  # making kernel instance
        accuracy = obj.ModelPredict(X_test, y_test, bias, kernelInstance) # call ModelPredict function
        return accuracy


    # check the validity of kernel name
    else:
        logging.error("Kernel name is not valid")
        sys.exit()

