import cvxopt
import cvxopt.solvers
cvxopt.solvers.options['show_progress'] = False
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import sys
import logging
import numpy as np
import sys
import math
from datetime import datetime
from decimal import Decimal
from .kernels import Jacobi,Legendre,Gegenbauer,Chebyshev




def transformation(x, T=1):
    """
    Transformation function with transition parameter, transforms/normalizes data into the selected kernel space.
    
    Parameters
    ----------
    x : Array
        The set to be normalized or transfomrd to a new space.
    T : float, optional
        Fractional Order, if 0 < T < 1, then transformation applies on input data-set and
           if T=1, then normalization applies on to input data-set.
           The default is 1.

    Returns
    -------
    Array
        Normalized/Transformed data-set.

    """
    new_x = []
    min = np.min(x)
    max = np.max(x)
    for i in range(len(x)):
        new_x.append(2 * (((x[i] - min) / (max - min)) ** (T)) - 1)
        
    return np.asarray(new_x)


def _Support_Multipliers(array,n,order='asc'):

        """
        This fucntions determines the support vectors among Lagrange multipliers. Output is dependant to value of paramter 'n'.

        Parameters
        ----------
        array : numpy array
            The Langrange Multipliers.

        n : char , int , scientific notation
            if it's a charachter == "A", then the average method is choosed.
            if it's a positive integer, then its the number of support vectors, that should be choosed from Lagrange Multipliers.
                if it's greater than number of Lagrange Nultipliers then all of them will be choosed!
            if it's a number expressed using scientific notation (ex: 1e-5), that's the minimum treshold,
                all the support vectos are SV >=  1e-5


        Returns
        -------
        numpy array of Support Multipliers.

        """
        def _isscientific(n):
            if 'e' in n.lower():
                return True
            else:
                return False

        def _format_e(nn):
            a = '%E' % nn
            return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


        def _S_N(x):

            if "-" in x:
                txt_mod=(x.split("-",2)[-1])
                if "+" in txt_mod:
                    return int(x.split("+",2)[-1])
                else:
                    return -(int(txt_mod))
            elif "+" in x:
                    txt_mod=x.split("+",2)[-1]
                    return int(txt_mod)



        #select directly first n elements of LMs wether in asc order or desc order
        if isinstance(n,int):
            print("** Strictly",n," first elements of support vectors are selected!")
            if order not in ('asc','desc'):
                print("** Error: order should be 'asc' or 'desc'")    
                sys.exit()
            elif order=='asc':
                support_m = np.sort(array)[:n]                      # Support Multipliers
            elif order=='desc':
                support_m = np.sort(array)[::-1][:n]

        #if n is a scientific number, select LMs as Support vector if are greater than n
        elif _isscientific(n):
            print("** Scientific number",n,"is received as threshold for support vectors")
            support_m = array>float(n)

        elif str(n).lower()=='a':
            print("** Avegage support vector determiner selected!")
            sum=0
            for i in array:
                c=_S_N(str(_format_e(Decimal(i))))
                sum +=c
            sv_treshold = int(sum/len(array))
            print("** sv threshold: 10^", sv_treshold)
            support_m = array > 10 ** +sv_treshold
        return support_m
 
        
class SVM(object):
    def __init__(self,
                 kernel,
                 order,
                         
                         
                 T,
                 k_param1,
                 k_param2,
                 sv_determiner,
                 form,
                 C,
                 noise):
        self.kernel = kernel
        self.order = order
                              
                               
        self.T = T
        self.k_param1 = k_param1
        self.k_param2 = k_param2
        self.sv_determiner = sv_determiner
        self.form = form
        self.C = C
        self.noise = noise
        
        if self.C is not None:
            self.C = float(self.C)
            
    

    def fit(self, x_train, y_train, log=False):
        """
        The main fitting function, to invoke the cvxopt and solve the convext SVM equation.
            
        Parameters
        ----------
        x_train: numpy array
            training set without labels

        y_train: numpy array
            training set only the labels
        
        log: bool
            If True the print fitting infotmation

        Returns
        -------
        wieght vector: numpy array
        support vectors: numpy array
        bias: float
        kernel instance: object
        """
     
        #Gram Matrix
        n_samples, n_features = x_train.shape
        K = np.zeros((n_samples, n_samples))
        
        if self.kernel == "Legendre":
            kernel_ins = Legendre(self.order)
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = kernel_ins.kernel(x_train[i], x_train[j])
                    
        elif self.kernel == "Jacobi":
            kernel_ins = Jacobi (self.k_param1,self.k_param2,self.order,self.noise)
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = kernel_ins.kernel(x_train[i], x_train[j])
                    
        elif self.kernel == "Gegenbauer":
            kernel_ins = Gegenbauer(self.order,self.k_param1)
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = kernel_ins.kernel(x_train[i], x_train[j])
                    
        elif self.kernel == "Chebyshev": 
            kernel_ins = Chebyshev(self.order)
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = kernel_ins.kernel(x_train[i],x_train[j],self.form)

        y = y_train.astype(float)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)
        




        # If it's Hard Margin
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:       #if it's  Soft Margin, the regularization parameter "C" comes into play
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
        Support vectors have non zero lagrange multipliers
        3 Ways to choose support vectors from Langrange Multipliers(LMs)
            - First receiving the minimume treshold from user and choose all LMs that are greater than that
            - Second finding the average and choose half of LMs
            - Third user chooses a number "n" of them strictly, then MLs will be sorted
                in asc/desc order and first "n" number of the will be choosed as Support Vectors
        """

    
        support_m_indices = _Support_Multipliers(lagrange_multipliers,self.sv_determiner,order=None)   # Compute Support Vectors
        self.support_multipliers  = lagrange_multipliers[support_m_indices]
        self.support_vectors = x_train[support_m_indices]       # Support Vectors are point from input data coresponding to support multipliers indices
        self.support_vector_labels = y[support_m_indices]       # label of Support Vectors

        self._weights = self.support_multipliers                # set support multipliers to weights


        if log==True:
            print("** Lagrange multipliers:", self.support_multipliers)
            print("** Max is:", np.max(self.support_multipliers), "Min is:", np.min(self.support_multipliers))
            print("** solution status:", solution['status'])
            print("** K condition value:", np.linalg.cond(K))
            New_matrix = np.hstack((P, np.transpose(A)))
            New_matrix = np.hstack((np.transpose(G), New_matrix))
            print("** [P, A_transpose, G_transpose] condition value:", np.linalg.cond(New_matrix))
            print("** Rank A is:", np.linalg.matrix_rank(A), "Rank [P, A_transpose, G_transpose] is:", np.linalg.matrix_rank(New_matrix))
            print("** ",len(self.a),"support vectors are seleceted out of",n_samples,"points")


        """
        A prediction on support vectors with bias = 0 and computing the error gives the bias
        """
        ind = np.arange(len(lagrange_multipliers))[support_m_indices]  
        _bias=0
        for n in range(len(self.support_multipliers)):
            _bias += self.support_vector_labels[n] 
            _bias -= np.sum(self.support_multipliers * self.support_vector_labels * K[ind[n], support_m_indices])
            
        _bias /= len(self.support_multipliers)        
        #self._bias = np.mean([y_k - SVM.predict(self,x=x_k,b=0,kernel=kernel_ins) for (y_k,x_k) in zip(self.support_vector_labels,self.support_vectors)])
        return self._weights, self.support_vectors, _bias , kernel_ins


    def predict(self,x,b,kernel):
        """
        Prediction function

        Parameters
        ----------
        x : Array
            the input array to predict the corresponding labels.
                according to fitted model.
        b : Float
            The bias to be used for prediction.
        kernel : Object
            The kernel class.

        Returns
        -------
        Array
            The array of predicted labels bias added.

        """
        
        if self.kernel == 'linear':
            return np.dot(x, self._weights) + b
        
        else:
            
            y_predict = np.zeros(len(x))
            # print("y_predict)
            for i in range(len(x)):
                s = 0
                for a, sv_y, sv in zip(self.support_multipliers, self.support_vector_labels, self.support_vectors):
                    s += a * sv_y * kernel.kernel(x[i], sv)     
                y_predict[i] = s
                
            return np.sign(y_predict + b)

class Model(object):
    """
        The main class that should be initiated first. This class initiates the requested model and parameters.
        Creating an object of this class by passing relevant parameters outcomes an object ready to be use din fitting procedure of a data-set.
        

                            
    """
    def __init__(self,
                 kernel="Chebyshev",
                 order=2,
                 T = 1,
                 param1 = None,
                 param2 = None,
                 svd = 'a',
                 form = 'r',
                 c=None,
                 noise=0.1,
                 orsvmModel=None):
        """
        
        Parameters
        ----------
        
        kernel : string 
            Kernel name, can be one of: Legendre, Jacobi, Gegenbauer, Chebyshev. default is Chebyshev.
            
        order: int
            Orthogonal kernel's order.
            
        T: float
            the Transition order. 0<T<1. default is 1, which means transition to normal space (normalization) instead of transformation.
            
        param1 : float, optional
            Default is None. 
            First hyperparameter of the kernels for Jacobi and Gegenbauer 
            Jacobi: param1 refers to psi, first hyper parameters of jacobi kernel.
            Gegenbauer: param1 refers to lambda, the hyperparameter of Gegenbauer kernel.
            
            
        param2: float, optional
            Default is None. 
            Second hyperparameter of the kernels for Jacobi and Gegenbauer 
            Jacobi: param2 refers to the omega, second hyperparameter of the jacobi kernel.
            
        
        svd : char , int , scientific number , optional
            The default is 'a'.
                         is the support vector determiner. 3 approaches are possible to choose the support vectors 
                         among Langrange Multipliers:  
                             1-  sv_determiner = int  :  User chooses how many of Lagrange multipliers, consider to be 
                                 support vectors.
                                 
                             2-  sv_determiner = 1e-3 :  User may choose a minimun threshold to choose support vectors from  
                                 Lagrange multipliers. In this case, the passed argument must be in scientific notation
                                 for example 1e-3, chooses all Lagrange multipliers that are greater than '1e-3'(0.001)
                                 
                             3-  sv_determiner = 'a'  :  (default) setting the sv_determiner = 'a'  and OSVM will compute the average 
                                 of scientific notation of the scale that Lagrange Multipliers are in, and sets as the 
                                 criteria to select the ones that are greater than average. Choosing the "average" may 
                                 cause most of Lagrange Multipliers to be selected as Support Vectors!
                                 And this may lead to poor generalization of the fitting result! but the benefit of this
                                 one is whenever the user may not know how to choose the threshold, in this case, choosing 
                                 the wrong threshold may outcomes 0 Support Vectors, therefore setting sv_determiner = 'a' for the
                                 first fit gives an intuition to choose the best value.
        form : CHAR, optional
            The default is 'r'. Possible values : 'r' or 'e'
                          which chooses how the Chebyshev kernel to be used wether explicit or recursive.
                          Only is applicable to Chebyshev kernel, for other kernels if paased, will be ignored.
                          
        C : int, optional
            Is the regulization parametr is SVM algorithm
            The default is None. Possible values: 10, 100, 1000 ...
        
        noise: float
            A noise  only applicable to weight function of Jacobi kernel. recommended values can be 0.1, 0.01,...
        """
        self.Kernel = kernel
        self.Order = order
        self.T = T
        self.KernelParam1 = param1
        self.KernelParam2 = param2
        self.SupportVectorDeterminer = svd
        self.Form = form
        self.C = c
        self.Noise = noise
        self.OrsvmModel = orsvmModel

    def ModelFit(self,
            x_train,
            y_train):
        """
            This is fit function of Model class.
            The dataset has to be provided into two sets as train and test set.
            Mandatory parameters are as follow:

            Parameters
            ----------
            x_train : Pandas dataframe
                DESCRIPTION. traning data
            
            y_train : Pandas data frame / numpy array
                DESCRIPTION. traning labels

            Returns
            -------
            Arrays of wieghts, Support vectors, and teh bias and initiated kernel instance
        """
        print("*"*10,datetime.now().strftime("%d/%m/%Y %H:%M:%S"),"*"*10)
        print("** OSVM kernel:",self.Kernel)
        print("** Order:",self.Order)
        if self.T < 1:
            print("** Fractional mode, transition :",self.T)
        else:
            print("** Normal Mode, Transition:",self.T)
        x_train = transformation(x_train, self.T)

        # Create the an object of SVM class with proper paramters with orthogonal kenrel
        self.OrsvmModel = SVM(self.Kernel, self.Order, self.T, self.KernelParam1, self.KernelParam2, self.SupportVectorDeterminer, self.Form, self.C,self.Noise)

        # Capture the Wieghts, Support Vectors and the Kernel itself for furthure use                                                   
        WeightVector,SupportVectors,Bias, kernelInstance = self.OrsvmModel.fit(x_train, y_train)       
        
        return WeightVector,SupportVectors,Bias,kernelInstance          # Fit the data and return the result

    def ModelPredict(self,
                x_test,
                y_test,
                bias,
                k_ins):
        """ Predict function of MOdel class. Using test-set returns the accuracy score of the model.
            Parameters
            ----------
            x_test : array/ DataFrame, optional
                Test set.
                
            y_test : array, optional
                Test set labels
            bias : float
                the bias number of hyperplane's equation. 
            k_ins : obj
                fitted kernel model.

             Returns
            -------
                prints the accuracy score, classification_report, confusion_matrix and returns the accuracy score
        """
        x_test = transformation(x_test, self.T)                                         # Transform the test-set into new space
        y_predict = self.OrsvmModel.predict(x_test,bias,k_ins)                          # find the accuracy
        print("** Accuracy score:",accuracy_score(y_test, y_predict))                   # Accuracy score 
        print("** Classification Report:\n",classification_report(y_test, y_predict))   # Classification Report      
        print("** Confusion Matrix:\n",confusion_matrix(y_test, y_predict))             # Confusion Matrix
        return accuracy_score(y_test, y_predict)                                        # Return the accuracy score
