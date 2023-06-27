import numpy as np
from .base import BaseBlock,InputWarpper,Timer
from .simulator import DynamicSystem
def to_list(x):
    if isinstance(x,np.ndarray):
        return x.reshape(-1).tolist()
    else:
        return x
class Transfer(BaseBlock):
    '''
    Transfer Function:
    Y(s) = [num]/[den]*U(s)
    init:
    y = dy = ddy...=0
    x = x
    '''
    def __init__(self,num,den,x=0,timer=Timer(), u = None):
        self.num = num
        self.den = den
        self.timer = timer# if isinstance(timer,Timer) else Timer(timer)
        self._x = x
        self._u = InputWarpper(u)
        self._y = x
    def init(self,x):
        self._x = x
        self.timer.init()
    def __iter__(self):
        self.init(0)
        for t in self.timer:
            yield t
    @property
    def u(self):
        pass
    @u.setter
    def u(self,u):
        self._u = InputWarpper(u)
    @property
    def y(self):
        pass
    def update(self, x):
        return self.y