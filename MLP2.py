from scipy.special import expit
import numpy as np
import random

class MLP:
    def __init__(self, MLP_config):
        self.alpha = MLP_config['alpha']
        self.Emin = MLP_config['Emin']
        self.input_layer_size  = MLP_config['input_size']
        self.hidden_layer_size = MLP_config['hidden_size']
        random.seed(MLP_config['seed'])

        self.y  = np.asarray([0.0 for _ in range(self.hidden_layer_size)])
        
        self.wi = np.asarray([[random.uniform(-0.5, 0.5) for _ in range(self.input_layer_size)] for _ in range(self.hidden_layer_size)])
        self.wj = np.asarray([random.uniform(-0.5, 0.5) for _ in range(self.hidden_layer_size)])

        self.Ti = np.asarray([random.uniform(-0.5, 0.5) for _ in range(self.hidden_layer_size)]) 
        self.Tj = np.asarray(random.uniform(-0.5, 0.5))

        self.output = 0  

    def feedforward(self, X):
        self.y = expit(np.subtract(np.dot(self.wi, X), self.Ti))

        return np.dot(self.wj, self.y) - self.Tj

    def backward(self, X, e):
        self.output_error = self.output - e

        self.hidden_error = self.wj * self.output_error
    
        self.wj -= self.alpha * self.output_error * self.y
        self.Tj += self.alpha * self.output_error

        gradient_i = self.hidden_error * (self.y * (1 - self.y))
        self.wi   -= self.alpha * np.outer(gradient_i, X)
        self.Ti   += self.alpha * gradient_i

    def cost_function(self, x_windows, e):
        err = 0
        for i in range(len(x_windows)):
            err += (self.feedforward(x_windows[i]) - e[i]) ** 2

        return err * 0.5    

    def absolute_error(self, x_windows, e):
        err = 0
        for i in range(len(x_windows)):
            y = self.feedforward(x_windows[i])
            err += abs(y - e[i]) / ((y + e[i]) / 2) * 100
        return err / len(x_windows)        

    def learning(self, x_windows, e, text):
        E = self.Emin
        while(E >= self.Emin):
            y_set = []
            E = 0
            for i in range(len(x_windows)):
                X = np.asarray(x_windows[i])
                
                self.output = self.feedforward(X)
                y_set.append(self.output)

                self.backward(X, e[i])

                E = self.cost_function(x_windows, e)
                print(text + str(E))  
                if E < self.Emin:
                    break

        y = []
        for i in x_windows:
            X = np.asarray(i)
            y.append(self.feedforward(X))

        return y

    def forecast(self, x_windows):
        y = []

        for i in x_windows:
            X = np.asarray(i)
            y.append(self.feedforward(X))

        return y
