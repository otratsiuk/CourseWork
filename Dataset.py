import numpy as np
import matplotlib.pyplot as plt
from ODESolver import ODESolver

class Dataset:
    dt_splitter = 2 / 3
    left_bound = -20
    right_bound = 20
    dt = 0.02
    initial = 0.1

    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.t = np.arange(self.left_bound, self.right_bound, self.dt)
        self.dt_size = len(self.t)

        self.training_dt_size = round(self.dt_size * self.dt_splitter)
        self.validation_dt_size = self.dt_size - self.training_dt_size

        solver = ODESolver(self.dx, self.dy, self.dz)
        solver.set_initial_conditions(self.initial, self.initial, self.initial)
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
