
class ODESolver:
    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.x = []
        self.y = []
        self.z = []

    def set_initial_conditions(self, x0, y0, z0):
        self.x.append(x0)
        self.y.append(y0)
        self.z.append(z0)

    def solve(self, n, dt):
        self.n = n

        for i in range(self.n - 1):
            Kx1 = dt * self.dx(self.x[i], self.y[i], self.z[i])
            Ky1 = dt * self.dy(self.x[i], self.y[i], self.z[i])
            Kz1 = dt * self.dz(self.x[i], self.y[i], self.z[i])

            Kx2 = dt * self.dx(self.x[i] + 0.5 * Kx1, self.y[i] + 0.5 * Ky1, self.z[i] + 0.5 * Kz1)
            Ky2 = dt * self.dy(self.x[i] + 0.5 * Kx1, self.y[i] + 0.5 * Ky1, self.z[i] + 0.5 * Kz1)
            Kz2 = dt * self.dz(self.x[i] + 0.5 * Kx1, self.y[i] + 0.5 * Ky1, self.z[i] + 0.5 * Kz1)

            Kx3 = dt * self.dx(self.x[i] + 0.5 * Kx2, self.y[i] + 0.5 * Ky2, self.z[i] + 0.5 * Kz2)
            Ky3 = dt * self.dy(self.x[i] + 0.5 * Kx2, self.y[i] + 0.5 * Ky2, self.z[i] + 0.5 * Kz2)
            Kz3 = dt * self.dz(self.x[i] + 0.5 * Kx2, self.y[i] + 0.5 * Ky2, self.z[i] + 0.5 * Kz2)

            Kx4 = dt * self.dx(self.x[i] + Kx3, self.y[i] + Ky3, self.z[i] + Kz3)
            Ky4 = dt * self.dy(self.x[i] + Kx3, self.y[i] + Ky3, self.z[i] + Kz3)
            Kz4 = dt * self.dz(self.x[i] + Kx3, self.y[i] + Ky3, self.z[i] + Kz3)

            self.x.append(self.x[i] + (1 / 6) * (Kx1 + 2 * Kx2 + 2 * Kx3 + Kx4))  
            self.y.append(self.y[i] + (1 / 6) * (Ky1 + 2 * Ky2 + 2 * Ky3 + Ky4))
            self.z.append(self.z[i] + (1 / 6) * (Kz1 + 2 * Kz2 + 2 * Kz3 + Kz4))

        return self.x, self.y, self.z







