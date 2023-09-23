import numpy as np
import matplotlib.pyplot as plt
from .base import InputWrapper, BaseBlock, Timer


class StateSpace(BaseBlock):
    """
    x_dot = A * x + B * u\n
    y = C * x + D * u
    """

    def __init__(self, A, B, C, D, timer=Timer(), x=None, u=None):
        super().__init__()
        self._A = A
        self._B = B
        self._C = C
        self._D = D
        self._x = x
        self.timer = timer
        self._u = InputWrapper(0) if u is None else InputWrapper(u)
        self.init_state = x
        # Save state of u,used to self.y
        # Once self.u is get ,update it
        self.u_temp = self.u

    @staticmethod
    def from_system(system, timer=Timer(), x=None, u=None):
        return StateSpace(system.A, system.B, system.C, system.D, timer, x, u)

    @property
    def u(self):
        # print("U")
        # return self._u.u
        # Update once u update.Capture state of u
        self.u_temp = np.array(np.array(self._u.y)).reshape((-1, 1))
        return np.array(np.array(self._u.y)).reshape((-1, 1))

    @u.setter
    def u(self, u):
        self.set_u(u)
        self.u_temp = self.u  # update self.u_temp

    def init(self, x=None):
        self._x = x
        self.init_state = x
        self.timer.init()

    # reset system state for last init() state
    def reset(self):
        self.init(self.init_state)

    def __iter__(self):
        # while True:
        #     yield self.update(self.u)
        #     if self.t>=self.t_stop:
        #         break
        self.timer.init()
        for t in self.timer:
            yield self.update_self()

    def simulate(self, draw=True):
        h = []
        for y in self:
            h.append([self.t, y.item()])
        h = np.array(h)
        if draw:
            plt.plot(h[:, 0], h[:, 1])
            plt.show()
        return h.copy()

    @property
    def x(self):
        return self._x

    @property
    def t(self):
        return self.timer._t

    @property
    def Ts(self):
        return self.timer.step

    @property
    def y(self):
        return self._C @ self.x + self._D @ self.u_temp

    # update x and u(to u_temp),if u is None,use self.u
    def update(self, u=None):
        try:
            u.__getattribute__("y")
        except AttributeError:
            pass
        if u is not None:
            self.u = u
        return self.update_self()

    # update using self.u
    def update_self(self):
        # https://zhuanlan.zhihu.com/p/326930724
        # Rank 1:
        # x_dot = self._A@self.x + self._B*u
        # self.x+=x_dot*self.Ts
        # Rank 2:
        # u = np.array(u).reshape((-1,1))
        phi = (np.eye(*self._A.shape) + self._A * self.Ts + 1 / 2 * self._A @ self._A * self.Ts ** 2)
        self._x = phi @ self.x + self._B @ self.u * self.Ts
        # Or
        # self.x = phi@self.x + 1/2*(1+phi)@self._B*u*self.Ts
        # self.timer.update()
        # self.u #update u
        return self._C @ self.x + self._D @ self.u

    def set_t_stop(self, t):
        self.timer.stop = t
        return self

    def set_u(self, u):
        self._u = InputWrapper(u)


class DiscreteStateSpace():
    """
        x_k = A_k * x_k-1 + u_k\n
        z_k = C_k * x_k
    """

    @staticmethod
    def from_continous(system: StateSpace, x0):
        system.init(x0)
        A = (np.eye(*system._A.shape) + system._A * system.Ts + 1 / 2 * system._A @ system._A * system.Ts ** 2)
        B = system._B * system.Ts
        x0 = system.x
        return DiscreteStateSpace(x0, A, system._C, B, system._D)

    def __init__(self, x0, A0, C0, B0=None, D0=None):
        self._A = A0
        self._C = C0
        self._x = x0
        self._B = 1 if B0 is None else B0
        self._D = 0 if D0 is None else D0
        self._k = 0
        self.k_stop = 100
        self.u = None

    def init(self, x):
        self._x = x

    def update(self, uk):
        self._x = self._A @ self._x + uk
        zk = self._C @ self._x
        self._k += 1
        return zk

    def __iter__(self):
        while True:
            yield self.update(self.u)
            if self._k >= self.k_stop:
                break
