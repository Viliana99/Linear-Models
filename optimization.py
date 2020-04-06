import numpy as np
import  oracles
import time


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    def __init__(self, loss_function, step_alpha=1, step_beta=0, tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        step_alpha - float, параметр выбора шага 
        step_beta- float, параметр выбора шага 
        tolerance - точность, по достижении которой, прекращаем оптимизацию.
       Критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 
        max_iter - максимальное число итераций     
        **kwargs - аргументы, необходимые для инициализации   
        """
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.oracle = oracles.BinaryLogistic(**kwargs)

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        trace - переменная типа bool
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        w = w_0
        last_func = self.oracle.func(X, y, w)
        if trace:
            history = {'func' : [last_func], 'time' : [0]}
        for step in range(1, self.max_iter + 1):
            start_time = time.time()
            last_func = self.oracle.func(X, y, w)
            nu_step = self.step_alpha / (step ** self.step_beta)
            w = w - nu_step * self.oracle.grad(X, y, w)
            new_func = self.oracle.func(X, y, w)
            if trace:
                history['time'].append(time.time() - start_time)
                history['func'].append(new_func)
            if abs(new_func - last_func) < self.tolerance:
                break
        self.weights = w
        if trace:
            return history
          
    def predict(self, X):
        """
        Получение меток ответов на выборке X
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: одномерный numpy array с предсказаниями
        """
        res = X.dot(self.weights)
        pred = np.ones_like(res)
        pred[np.where(res < 0)[0]] = -1
        return pred.astype(int)

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k 
        """
        p = expit(X.dot(self.weights))
        return np.array([p , 1 - p]).T     

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: float
        """
        return self.oracle.func(X, y, self.weights) 

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: numpy array, размерность зависит от задачи
        """
        return self.oracle(X, y, self.weights)

    def get_weights(self):
        """
        Получение значения весов функционала
        """    
        return self.weights


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0, tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
       """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        batch_size - размер подвыборки, по которой считается градиент        
        step_alpha - float, параметр выбора шага
        step_beta- float, параметр выбора шага
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 
        max_iter - максимальное число итераций (эпох)
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        **kwargs - аргументы, необходимые для инициализации
        """
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.oracle = oracles.BinaryLogistic(**kwargs)

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        Если trace = True, то метод  возвращают словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)
        w = w_0
        step = 1
        view_item = 0
        epoch = 0
        last_w = w
        last_func = self.oracle.func(X, y, w)
        shift = 0
        index_shuffle = np.random.permutation(X.shape[0])
        start_time = time.time()
        if trace:
            history = {'func' : [last_func], 'time' : [0], 'epoch_num' : [0], 'weights_diff': [0]}
        for step in range(1, self.max_iter + 1):
            if (X.shape[0] - shift) <= 0:
                index_shuffle = np.random.permutation(X.shape[0])
                shift = 0
            index = index_shuffle[shift : shift + self.batch_size]
            view_item += index.shape[0]
            shift += index.shape[0]
            last_func = self.oracle.func(X, y, w)
            nu_step = self.step_alpha / (step ** self.step_beta)
            w = w - nu_step * self.oracle.grad(X[index], y[index], w)
            new_func = self.oracle.func(X, y, w)
            if trace:
                if (view_item / X.shape[0] - epoch) > log_freq:
                    epoch = view_item / X.shape[0]
                    history['time'].append(time.time() - start_time)
                    history['func'].append(new_func)
                    history['epoch_num'].append(epoch)
                    history['weights_diff'].append((w - last_w).T.dot(w - last_w))
                    last_w = w
            if abs(new_func - last_func) < self.tolerance:
                break
        self.weights = w
        if trace:
            return history