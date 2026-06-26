import numpy as np
import copy 
import math
class BGDRegression:
    #ver 1: running batch gradient descent regression using partial vectorization to see the math clearer

    def compute_cost_loop(self, x, y, w, b):
        """
        Calculate cost
            x: numpy matrix
            y: target variable
            w (1D numpy array): weight 
            b (scalar): bias
        """
        m = x.shape[0]
        total_cost = 0
        for i in range(m):
            total_cost += (np.dot(x[i], w) + b - y[i]) ** 2
        total_cost /= 2 * m
        return total_cost
    
    def compute_gradient_loop(self, x, y, w, b):
        """
        Calculate the gradient of the weights for the next iteration 
            x: numpy matrix
            y: target variable
            w (1D numpy array): weight 
            b (scalar): bias 
        """
        m, n = x.shape
        df_dw = np.zeros((n,))
        df_db = 0.0
        for i in range(m):
            error = np.dot(x[i], w) + b - y[i]
            for j in range(n):
                df_dw[j] += error * x[i, j]
            df_db += error
        return df_dw / m, df_db / m
    
    def run_bgd_regressor_loop(self, x, y, w, b, alpha, iterations):
        """
        run batch gradient descent to find local minimum point of J(w,b) 
            x: numpy matrix
            y: target variable
            w (1D numpy array): weight 
            b (scalar) : bias 
            alpha (float): learning rate
            iterations (int)
        return w, b, cost_history[], w,b_history[]
        """
        w_local = copy.deepcopy(w)
        b_local = b
        J_history = []
        wb_hist = []
        for i in range(iterations):
            df_dw, df_db = self.compute_gradient_loop(x, y, w_local, b_local)
            w_local -= alpha * df_dw 
            b_local -= alpha * df_db
            J_history.append(self.compute_cost_loop(x, y, w_local, b_local))
            wb_hist.append((w_local, b_local))
            if (i%(math.ceil(iterations / 10)) == 0):
                print(f"iteration {i}: cost = {J_history[-1]:.4f}")
        return w_local, b_local, J_history, wb_hist
    
    #ver 2: running batch gradient descent regression  fully vectorized
    def compute_cost(self, x, y, w, b):
        """
        Calculate cost
            x: numpy matrix
            y: target variable
            w (1D numpy array): weight 
            b (scalar): bias
        """
        m = x.shape[0]
        error = (np.dot(x, w) + b - y) ** 2
        total_error = np.sum(error)
        total_error /= 2 * m
        return total_error
    def compute_gradient(self, x, y, w, b):
        """
        Calculate the gradient of the weights for the next iteration 
            x: numpy matrix
            y: target variable
            w (1D numpy array): weight 
            b (scalar): bias 
        """
        m = x.shape[0]
        error = np.dot(x, w) + b - y
        df_dw = np.dot(x.T, error) / m
        df_db = np.sum(error) / m
        return df_dw, df_db
    def run_bgd_regressor(self, x, y, w, b, alpha, iterations):
        """
        run batch gradient descent to find local minimum point of J(w,b) 
            x: numpy matrix
            y: target variable
            w (1D numpy array): weight 
            b (scalar) : bias 
            alpha (float): learning rate
            iterations (int)
        return w, b, cost_history[], w,b_history[]
        """
        w_local = copy.deepcopy(w)
        b_local = b
        J_history = []
        wb_hist = []
        for i in range(iterations):
            df_dw, df_db = self.compute_gradient(x, y, w_local, b_local)
            w_local -= alpha * df_dw 
            b_local -= alpha * df_db
            J_history.append(self.compute_cost(x, y, w_local, b_local))
            wb_hist.append((w_local, b_local))
            if (i%(math.ceil(iterations / 10)) == 0):
                print(f"iteration {i}: cost = {J_history[-1]:.4f}")
        return w_local, b_local, J_history, wb_hist
        