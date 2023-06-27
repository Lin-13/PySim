import numpy as np
import matplotlib.pyplot as plt
from .base import InputWarpper,BaseBlock,Timer
import random

class DynamicSystem():
    '''
    x_dot = A * x + B * u
    y = C * x + D * u
    '''
    
    def __init__(self,A,B,C,D,Ts = 0.001,t_stop = 10,x = None,u = None):
        self._A = A
        self._B = B
        self._C = C
        self._D = D
        self.x = x
        self.Ts = Ts
        self._u = InputWarpper(u)
        # for iter
        self.t = 0
        self.t_stop = t_stop
    @staticmethod
    def from_system(system,Ts = 0.001,t_stop = 10,x = None,u = None):
        return DynamicSystem(system.A,system.B,system.C,system.D,Ts,t_stop,x,u)
    @property
    def u(self):
        # print("U")
        return self._u.u
    @u.setter
    def u(self,u):
        self.set_u(u)
    def init(self,x):
        self.x = x
        self.t = 0
    def __iter__(self):
        while True:
            yield self.update(self.u)
            if self.t>=self.t_stop:
                break
    def update(self,u):
        try:
            u = u.__getattribute__("u")
        except AttributeError:
            pass
        # if u == None and self.u ==None:
        #     raise SyntaxError("PLease set or input u!")
        # https://zhuanlan.zhihu.com/p/326930724
        #Rank 1:
        # x_dot = self._A@self.x + self._B*u
        # self.x+=x_dot*self.Ts
        #Rank 2:
        phi = (np.eye(*self._A.shape)+self._A*self.Ts+1/2*self._A@self._A*self.Ts**2)
        self.x = phi@self.x + self._B*u*self.Ts
        # Or
        # self.x = phi@self.x + 1/2*(1+phi)@self._B*u*self.Ts
        self.t+=self.Ts
        return self._C@self.x + self._D*u
    def set_t_stop(self,t):
        self.t_stop = t
        return self
    def set_u(self,u):
        self._u = InputWarpper(u)
    

class DiscreteDynamicSystem():
    '''
        x_k = A_k * x_k-1 + u_k
        z_k = C_k * x_k
    '''
    @staticmethod
    def from_continous(system:DynamicSystem,x0):
        system.init(x0)
        A = (np.eye(*system._A.shape)+system._A*system.Ts+1/2*system._A@system._A*system.Ts**2)
        B = system._B*system.Ts
        x0 = system.x
        return DiscreteDynamicSystem(x0,A,system._C,B,system._D)
    def __init__(self,x0,A0,C0,B0 = None,D0 = None):
        self._A = A0
        self._C = C0
        self._x = x0
        self._B = 1 if B0 is None else B0
        self._D = 0 if D0 is None else D0
        self._k = 0
        self.k_stop = 100
        self.u = None
    def update (self,uk):
        self._x = self._A@self._x + uk
        zk = self._C@self._x
        self._k+=1
        return zk
    def __iter__(self):
        while True:
            yield self.update(self.u)
            if self._k>=self.k_stop:
                break
        
class TimeVariable():
    '''
    function warpper for varible
        y = fn(t,*fcn_args,**kwards)
    you can get y = varible.u if need
    Sample:
        u = TimeVariable(fn=TimeVariable.random,fcn_args = (1,2))
    Or define a function:
        def fcn(t,a,b):
            return a*t+b
    then
        u = variable(0,1,fn=fcn,a=1,b=2)
    or:
        u = variable(0,1,fn=fcn,fcn_args = [1,2])
    '''
    def __init__(self,t=0,step=0.01,stop = 10,fn = None,**kwards) -> None:
        self.t = t
        self.step = step
        self.stop = stop
        self.fcn = fn
        self.fcn_args = None
        self.fcn_kwards = {}
        try:
            self.fcn_args = kwards["fcn_args"]
        except KeyError:
            self.fcn_kwards = kwards
    def set_t(self,t):
        self.t = t
    def __iter__(self):
        self.set_t(0)
        while True:
            yield self.u
            if self.t>self.stop:
                break
    @staticmethod
    def random(t,*args,**kwards):
        return np.random.randn(*args)
    @property
    def u(self):
        ret = self.t
        if self.fcn ==None:
            return  self.t
        if self.fcn_args is not None:
            ret =  self.fcn(self.t,*self.fcn_args,**self.fcn_kwards)
        else:
            ret =  self.fcn(self.t,**self.fcn_kwards)
        self.set_t(self.t+self.step)
        return ret
class WhiteNoise(TimeVariable):
    '''
        return randn noise with ndarray.
        N:length of White Noise ,used to iter
    '''
    def __init__(self,mu = 0,sigma = 0.01,N = 100,shape = (1,)):
        self.shape = shape
        self.mu = np.asarray(mu) if isinstance(mu,np.ndarray) else np.array([mu])
        self.sigma = np.asarray(sigma) if isinstance(sigma,np.ndarray) else np.array([sigma])
        super().__init__(0,1,N,WhiteNoise.random,fcn_args = self.shape)
    def __iter__(self):
        self.set_t(0)
        while True:
            yield self.u
            if self.t>self.stop:
                break
    @property
    def u(self):
        # return TimeVariable.random(0,*self.shape)*self.sigma+self.mu
        return super().u*self.sigma+self.mu
