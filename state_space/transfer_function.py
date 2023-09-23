import numpy as np
from .base import BaseBlock, InputWrapper, Timer
from .state_space import StateSpace


def to_tuple(x):
    if isinstance(x, np.ndarray):
        return tuple(x.reshape(-1).tolist())
    elif isinstance(x, list) or isinstance(x, tuple):
        return tuple(x)
    else:
        raise TypeError


def get_state_matrix(num: tuple, den: tuple):
    m, n = len(num) - 1, len(den) - 1
    E = np.eye(n, n)
    E[-1, :] = np.array(den[-2::-1])
    A = np.zeros((n, n))
    for raw in range(n):
        if raw == n - 1:
            A[raw, 0] = -den[-1]
        else:
            A[raw, raw + 1] = 1
    E_inv = np.linalg.inv(E)
    A = E_inv @ A
    B = np.zeros((n, m + 1))
    B[-1, :] = np.array(num[-1::-1])
    B = E_inv @ B
    C = np.zeros((1, n))
    C[0, 0] = 1
    D = np.zeros((1, m + 1))
    return A, B, C, D


class Transfer(StateSpace):
    """
    @Brief:
    Transfer Function:\n
    Y(s) = [num]/[den]*U(s)\n
                (b_m*s**m+b_(m-1)*s**(m-1) + ... +b_0)
         = ---------------------------------------------- * U(s)
                (a_n*s**n+a_(n-1)*s**(n-1) + ... +a_0)

    work as State-Space Eq as .state_space.StateSpace
        X_dot = A*X + B*u\n
        y = C*X + D*u\n
        get Matrix by function get_state_matrix(num,den) in __init__\n
    with:
        X = [y' y'' ... y(n-1)].T

        y = y

        u = [u <history information in Transfer u>].T

    init:\n
    x (y in __init__) must b an array of shape (n,1)
    """

    def __init__(self, num, den, y=0, timer=Timer(), u=None):
        self._num = to_tuple(num)
        self._den = to_tuple(den)
        matrix = get_state_matrix(num, den)
        # x is X of state space
        x_ = np.zeros((len(den) - 1, 1))
        self.init_state = y
        if not isinstance(y, np.ndarray):
            x_[0, 0] = y
            x = x_
        else:
            x = y

        self._last_u = 0
        if u is None:
            self._u_0 = InputWrapper(0)
        else:
            self._u_0 = InputWrapper(u)
        self._du = np.zeros((len(self._num) - 1, 1))

        super().__init__(*matrix, timer, x, u)

        self._y = x

    def init(self, x=None):
        self.init_state = x
        x_ = np.zeros((len(self._den) - 1, 1))
        if not isinstance(x, np.ndarray):
            x_[0, 0] = x
            x = x_
        self._x = x
        self.timer.init()

    # def __iter__(self):
    #     for t in self.timer:
    #         yield t
    def reset(self):
        self.init(self.init_state)

    @property
    def num(self):
        return self._num

    @property
    def den(self):
        return self._den

    @property
    def u(self):
        self.u_temp = np.concatenate((np.array([[self._u_0.y]]).reshape((-1, 1)), self._du))
        return np.concatenate((np.array([[self._u_0.y]]).reshape((-1, 1)), self._du))

    # @u.setter in DynamicSystem
    def set_u(self, u):
        if isinstance(u, np.ndarray):
            u = u[0]
        self._u_0 = InputWrapper(u)
        self._u = self._u_0
        self._du = np.zeros((len(self._num) - 1, 1))

    @u.setter
    def u(self, u):
        self.set_u(u)
        self.u_temp = self.u  # update self.u_temp

    # @property
    # def y(self):
    #     return self._C@self.x + self._D@self.u
    def update(self, u=None):
        if u is not None:
            self.u = u
        # fix:super().update(self,u[0])
        # super().update(self,u[0])
        super().update()
        return self.y


def conv(a, b):
    assert (isinstance(a, tuple) and isinstance(b, tuple))
    m, n = len(a) - 1, len(b) - 1
    k = m + n
    c = [0] * (k + 1)
    for i in range(len(a)):
        for j in range(len(b)):
            c[i + j] += a[i] * b[j]
    return tuple(c)


class Gain(BaseBlock):
    """
        you can use gain to connect any system based on BaseBlock\n
        Or it has attribute "y"\n
        Warning:it does not save the state of u like Transfer or Dynamic\n
        it runs like self.y = self.k * self.u
    """

    def __init__(self, K, u=None):
        super().__init__()
        self.K = K
        self._u = InputWrapper(0) if u is None else InputWrapper(u)

    def init(self, x=None):
        pass

    @property
    def u(self):
        return self._u.y

    @u.setter
    def u(self, u):
        self.set_u(u)

    def set_u(self, u):
        self._u = InputWrapper(u)

    @property
    def y(self):
        return self.K * self.u

    def update(self, u=None):
        pass


def add(a, b):
    assert (isinstance(a, tuple) and isinstance(b, tuple))
    if len(a) > len(b):
        c = (0,) * (len(a) - len(b))
        b = c + b
    else:
        c = (0,) * (len(b) - len(a))
        a = c + a
    c = []
    for i in range(len(a)):
        c.append(a[i] + b[i])
    return tuple(c)


def sub(a, b):
    assert (isinstance(a, tuple) and isinstance(b, tuple))
    if len(a) > len(b):
        c = (0,) * (len(a) - len(b))
        b = c + b
    else:
        c = (0,) * (len(b) - len(a))
        a = c + a
    c = []
    for i in range(len(a)):
        c.append(a[i] - b[i])
    return tuple(c)


# Create a new Transfer by a and b
def serial(a, b, timer=Timer()):
    assert (isinstance(a, Transfer) and isinstance(b, Transfer))
    num1, den1 = a.num, a.den
    num2, den2 = b.num, b.den
    num, den = conv(num1, num2), conv(den1, den2)
    return Transfer(num, den, timer=timer)


# Default negative feedback
def feedback(G, H, timer=Timer(), negative=True):
    assert (isinstance(G, Transfer) and isinstance(H, Transfer))
    num1, den1 = G.num, G.den
    num2, den2 = H.num, H.den
    num = conv(num1, den2)
    if negative:
        den = add(conv(den1, den2), conv(num1, num2))
    else:
        den = sub(conv(den1, den2), conv(num1, num2))
    return Transfer(num, den, timer=timer)


if __name__ == "__main__":
    F = Transfer((1,), (1, 1))
    print(F.A, F._B, F._C, F._D)
