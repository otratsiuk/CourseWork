import math
import matplotlib.pyplot as plt
from scipy.special import expit
import numpy as np
import random
from Dataset import Dataset
from multiprocessing.pool import ThreadPool

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
        self.wi    -= self.alpha * np.outer(gradient_i, X)
        self.Ti    += self.alpha * gradient_i

    def cost_function(self, x_windows, e):
        err = 0
        for i in range(len(x_windows)):
            err += (e[i] - self.feedforward(x_windows[i])) ** 2

        return err * 0.5    

    def learning(self, x_windows, e, t, text):
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
        t = dataset.training_time_points()
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

def dx(x, y, z):
    return -y - z


def dy(x, y, z):
    return x + 0.1 * y


def dz(x, y, z):
    return 0.1 + z * (x - 5)

def plotter(t, y, e, title):
    plt.title(title)
    plt.plot(t[: len(y)], y, "red", label='forecast')
    plt.plot(t[: len(y)], e, "blue", label='original')
    plt.legend()
    plt.show()

MLP_config_x = {'alpha': 0.001, 'Emin': 0.1, 'input_size': 16, 'hidden_size': 50, 'seed': 67}
MLP_config_y = {'alpha': 0.001, 'Emin': 0.1, 'input_size': 16, 'hidden_size': 50, 'seed': 67}
MLP_config_z = {'alpha': 0.01, 'Emin': 0.001, 'input_size': 16, 'hidden_size': 50, 'seed': 67}

mlp_x = MLP(MLP_config_x)
mlp_y = MLP(MLP_config_y)
mlp_z = MLP(MLP_config_z)

dataset = Dataset(dx, dy, dz)
xt, yt, zt = dataset.training_sample()
x_windowst, ext = dataset.sliding_window_samples(xt, MLP_config_x['input_size'])
y_windowst, eyt = dataset.sliding_window_samples(yt, MLP_config_y['input_size'])
z_windowst, ezt = dataset.sliding_window_samples(zt, MLP_config_z['input_size'])
t_training = dataset.training_time_points()

xf, yf, zf = dataset.forecasting_sample()
x_windowsf, exf = dataset.sliding_window_samples(xf, MLP_config_x['input_size'])
y_windowsf, eyf = dataset.sliding_window_samples(yf, MLP_config_y['input_size'])
z_windowsf, ezf = dataset.sliding_window_samples(zf, MLP_config_z['input_size'])
t_forecasting = dataset.forecasting_time_points()

pool1 = ThreadPool(processes=1)
pool2 = ThreadPool(processes=1)
pool3 = ThreadPool(processes=1)

async_result1 = pool1.apply_async(mlp_x.learning, (x_windowst, ext, t_training, '[X]: '))
async_result2 = pool2.apply_async(mlp_y.learning, (y_windowst, eyt, t_training, '[Y]: '))
async_result3 = pool3.apply_async(mlp_z.learning, (z_windowst, ezt, t_training, '[Z]: '))

xt = async_result1.get()
xf = mlp_x.forecast(x_windowsf)

yt = async_result2.get()
yf = mlp_y.forecast(y_windowsf)

zt = async_result3.get()
zf = mlp_z.forecast(z_windowsf)

plotter(t_training, xt, ext, 'Training interval X(t)')
plotter(t_forecasting, xf, exf, 'Forecasting interval X(t)')

plotter(t_training, yt, eyt, 'Training interval Y(t)')
plotter(t_forecasting, yf, eyf, 'Forecasting interval Y(t)')

plotter(t_training, zt, ezt, 'Training interval Z(t)')
plotter(t_forecasting, zf, ezf, 'Forecasting interval Z(t)')

ax = plt.axes(projection='3d')
ax.plot(ext, eyt, ezt, 'blue')
ax.plot(xt, yt, zt, 'red')
plt.show()

ax = plt.axes(projection='3d')
ax.plot(exf, eyf, ezf, 'blue')
ax.plot(xf, yf, zf, 'red')
plt.show()