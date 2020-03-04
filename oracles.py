import numpy as np
import scipy
from scipy.special import expit 
from scipy.sparse import csr_matrix

class BaseSmoothOracle:

    """

    Базовый класс для реализации оракулов.

    """

    def func(self, w):

        """

        Вычислить значение функции в точке w.

        """

        raise NotImplementedError('Func oracle is not implemented.')



    def grad(self, w):

        """

        Вычислить значение градиента функции в точке w.

        """

        raise NotImplementedError('Grad oracle is not implemented.')



        

class BinaryLogistic(BaseSmoothOracle):

    """

    Оракул для задачи двухклассовой логистической регрессии.

    

    Оракул должен поддерживать l2 регуляризацию.

    """

    

    def __init__(self, l2_coef):

        """

        Задание параметров оракула.

        

        l2_coef - коэффициент l2 регуляризации

        """
        
        self.l2_coef = l2_coef

     

    def func(self, X, y, w=None):

        """

        Вычислить значение функционала в точке w на выборке X с ответами y.

        

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        

        y - одномерный numpy array

        

        w - одномерный numpy array

        """
        if w is None:
            w = np.zeros(X.shape[1])  
        marg = -y * X.dot(w)
        reg = self.l2_coef * w.T.dot(w) / 2
        loss = np.logaddexp(0, marg).mean()
        return loss + reg

        

    def grad(self, X, y, w=None):

        """

        Вычислить градиент функционала в точке w на выборке X с ответами y.

        

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        

        y - одномерный numpy array

        

        w - одномерный numpy array

        """
        if w is None:
            w = np.zeros(X.shape[1])
        marg = -y * X.dot(w)
        reg = self.l2_coef * w
        if isinstance(X, np.ndarray):
            grad = ((-y * expit(marg))[:,np.newaxis] * X).mean(axis = 0)
        elif isinstance(X, csr_matrix):
            grad = np.array((X.multiply((-y * expit(marg))[:,np.newaxis])).mean(axis = 0)).ravel()
        return grad + reg