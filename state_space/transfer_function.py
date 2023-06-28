import numpy as np
from .base import BaseBlock,InputWarpper,Timer
from .simulator import DynamicSystem
def to_tuple(x):
    if isinstance(x,np.ndarray):
        return tuple(x.reshape(-1).tolist())
    elif isinstance(x,list) or isinstance(x,tuple):
        return tuple(x)
    else:
        raise TypeError
def get_state_matrix(num:tuple,den:tuple):
    m,n = len(num)-1,len(den)-1
    E = np.eye(n,n)
    E[-1,:] = np.array(den[-2::-1])
    A = np.zeros((n,n))
    for raw in range(n):
        if raw ==n-1:
            A[raw,0] = -den[-1]
        else:
            A[raw,raw+1] = 1
    E_inv = np.linalg.inv(E)
    A = E_inv@A
    B = np.zeros((n,m+1))
    B[-1,:] = np.array(num[-1::-1])
    B = E_inv@B
    C = np.zeros((1,n))
    C[0,0] = 1
    D = np.zeros((1,m+1))
    return (A,B,C,D)
class Transfer(DynamicSystem):
    '''
    Transfer Function:
    Y(s) = [num]/[den]*U(s)
          (b_m*s**m+b_(m-1)*s**(m-1) + ... +b_0)/
        = --------------------------------------- * U(s)
          (a_n*s**n+a_(n-1)*s**(n-1) + ... +a_0)
    work as State-Space Eq as .simulator.DynamicSystem
        X_dot = A*X + B*u
        y = C*X + D*u
        get Matrix by function get_state_matrix(num,den) in __init__
    with:
    X = [y y' y'' ... y(n-1)].T
    y = y
    u = [u <history information in Transfer u>].T
    init:
    x must b an array of shape (n,1)
    '''
    def __init__(self,num,den,y=0,timer=Timer(), u = None):
        self._num = to_tuple(num)
        self._den = to_tuple(den)
        matrix = get_state_matrix(num,den)
        #x is X of state space 
        x_ = np.zeros((len(den)-1,1))
        if not isinstance(y,np.ndarray):
            x_[0,0] = y
            x = x_
        else:
            x = y
        super().__init__(*matrix,timer,x,u)
        # self.timer = timer# if isinstance(timer,Timer) else Timer(timer)
        # self._x = x
        self._last_u = 0
        if u is None:
            self._u_0 = InputWarpper(0)
        else:
            self._u_0 = InputWarpper(u)
        self._du = np.zeros((len(self._num)-1,1))
        self._y = x
    def init(self,x):
        self._x = x
        self.timer.init()
    # def __iter__(self):
    #     for t in self.timer:
    #         yield t
    def num(self):
        return self._num
    @property
    def den(self):
        return self._den
    @property
    def u(self):
        return np.concatenate((np.array([[self._u_0.y]]).reshape((-1,1)),self._du))
    # @u.setter in DynamicSystem
    def set_u(self,u):
        if isinstance(u,np.ndarray):
            u = u[0]
        self._u_0 = InputWarpper(u)
        self._du = np.zeros((len(self._num)-1,1))
    @u.setter
    def u(self,u):
        self.set_u(u)
    # @property
    # def y(self):
    #     return self._C@self.x + self._D@self.u
    def update(self, u):
        if self.t <self.Ts:
            self.u = u
        super().update(self.u[0,0])
        return self.y
if __name__ =="__main__":
    F = Transfer((1,),(1,1))
    print(F._A,F._B,F._C,F._D)