from MLP2 import MLP
from Dataset import Dataset
from multiprocessing.pool import ThreadPool
from datetime import datetime
import Plotter

def dx(x, y, z):
    return -y - z

def dy(x, y, z):
    return x + 0.1 * y

def dz(x, y, z):
    return 0.1 + z * (x - 5)

if __name__ == "__main__":
    input_size = 16

    MLP_config_x = {'alpha': 0.001, 'Emin': 0.1, 'input_size': 16, 'hidden_size': 50, 'seed': 67}
    MLP_config_y = {'alpha': 0.001, 'Emin': 0.1, 'input_size': 16, 'hidden_size': 50, 'seed': 67}
    MLP_config_z = {'alpha': 0.01, 'Emin': 0.001, 'input_size': 16, 'hidden_size': 50, 'seed': 67}

    mlp_x = MLP(MLP_config_x)
    mlp_y = MLP(MLP_config_y)
    mlp_z = MLP(MLP_config_z)

    dataset = Dataset(dx, dy, dz)
    t, x, y, z = dataset.full_sample()
    xt, yt, zt = dataset.training_sample()
    x_windowst, ext = dataset.sliding_window_samples(xt, MLP_config_x['input_size'])
    y_windowst, eyt = dataset.sliding_window_samples(yt, MLP_config_y['input_size'])
    z_windowst, ezt = dataset.sliding_window_samples(zt, MLP_config_z['input_size'])
    t_training = dataset.training_time_points(input_size)

    xf, yf, zf = dataset.forecasting_sample()
    x_windowsf, exf = dataset.sliding_window_samples(xf, MLP_config_x['input_size'])
    y_windowsf, eyf = dataset.sliding_window_samples(yf, MLP_config_y['input_size'])
    z_windowsf, ezf = dataset.sliding_window_samples(zf, MLP_config_z['input_size'])
    t_forecasting = dataset.forecasting_time_points(input_size)

    pool = ThreadPool(processes=3)

    start_time = datetime.now()

    async_result1 = pool.apply_async(mlp_x.learning, (x_windowst, ext, '[X]: '))
    async_result2 = pool.apply_async(mlp_y.learning, (y_windowst, eyt, '[Y]: '))
    async_result3 = pool.apply_async(mlp_z.learning, (z_windowst, ezt, '[Z]: '))

    xt = async_result1.get()
    yt = async_result2.get()
    zt = async_result3.get()

    print('MLPs training time: ' + str(datetime.now() - start_time))

    xf = mlp_x.forecast(x_windowsf)
    yf = mlp_y.forecast(y_windowsf)
    zf = mlp_z.forecast(z_windowsf)

    Plotter.show_dataset(t, x, y, z)

    Plotter.show_interval('approximation', t_training, xt, ext, yt, eyt, zt, ezt)
    Plotter.show_interval('forecast', t_forecasting, xf, exf, yf, eyf, zf, ezf)
