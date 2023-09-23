from .base import Timer, BaseBlock
import numpy as np


class Pendulum:
    def __init__(self, m, l, x=np.array([[0], [0]])):
        self.m = m
        self.g = 9.81
        self.l = l
        self.Ts = 0.01
        self.x = x
        self.x_init = x

    def init(self, x):
        self.x = x
        self.x_init = x

    def reset(self):
        self.x = self.x_init

    @property
    def A(self):
        return np.array([[0, 1], [-self.g / self.l, 0]])

    @property
    def B(self):
        return np.array([[0], [1 / (self.m * self.l ** 2)]])

    @property
    def C(self):
        return np.array([[1, 0]])

    @property
    def D(self):
        return np.array([0])

    def update(self, x):
        pass
