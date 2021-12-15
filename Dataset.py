import numpy as np
from ODESolver import ODESolver

class Dataset:
    dt_splitter = 2 / 3
    dt_size = 2000
    time_start = 0
    time_interval = 0.1
    dt = 0.02
    initial_condition = 0.1

    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.t = np.arange(self.time_start, self.dt_size * self.time_interval, self.time_interval)

        self.training_dt_size = round(self.dt_size * self.dt_splitter)
        self.validation_dt_size = self.dt_size - self.training_dt_size

        solver = ODESolver(self.dx, self.dy, self.dz)
        solver.set_initial_conditions(self.initial_condition, self.initial_condition, self.initial_condition)
        self.x, self.y, self.z = solver.solve(self.dt_size, self.dt)
            

    def training_sample(self):
        return self.x[: self.training_dt_size], self.y[: self.training_dt_size], self.z[: self.training_dt_size]

    def forecasting_sample(self):
        return self.x[self.training_dt_size :], self.y[self.training_dt_size :], self.z[self.training_dt_size :]

    def full_sample(self):
        return self.t, self.x, self.y, self.z    

    def forecasting_time_points(self, k):
        return self.t[self.training_dt_size : self.dt_size - k]  

    def training_time_points(self, k):
        return self.t[k : self.training_dt_size]        

    def sliding_window_samples(self, set, k):
        samples = []
        for i in range(len(set) - k):
            temp = []
            for j in range(k):
                temp.append(set[i + j])
            samples.append(temp)

        return samples, set[k :]  
