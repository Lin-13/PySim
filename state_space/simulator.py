import numpy as np
import matplotlib.pyplot as plt
from .base import InputWarpper,BaseBlock,Timer
import random

class DynamicSystem(BaseBlock):
    '''
    x_dot = A * x + B * u
    y = C * x + D * u
    '''
    
    def __init__(self,A,B,C,D,timer = Timer(),x = None,u = None):
        self._A = A
        self._B = B
        self._C = C
        self._D = D
        self._x = x
        self.timer = timer
        self._u = InputWarpper(u)
        #Save state of u
        self.u_temp = None
    @staticmethod
    def from_system(system,timer = Timer(),x = None,u = None):
        return DynamicSystem(system.A,system.B,system.C,system.D,timer,x,u)
    @property
    def u(self):
        # print("U")
        # return self._u.u
        #Update once u update.Capture state of u
        self.u_temp = np.array(np.array(self._u.y)).reshape((-1,1))
        return np.array(np.array(self._u.y)).reshape((-1,1))
    @u.setter
    def u(self,u):
        self.set_u(u)
        self.u #update self.u_temp
    def init(self,x):
        self._x = x
        self.timer.init()
    def __iter__(self):
        # while True:
        #     yield self.update(self.u)
        #     if self.t>=self.t_stop:
        #         break
        self.timer.init()
        for t in self.timer:
            yield self.update_self()
    def simulate(self,draw = True):
        h = []
        for y in self:
            h.append([self.t,y.item()])
        h = np.array(h)
        if draw:
            plt.plot(h[:,0],h[:,1])
            plt.show()
        return (h.copy())
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
        return self._C@self.x + self._D@self.u_temp
    def update(self,u):
        try:
            u = u.__getattribute__("y")
        except AttributeError:
            pass
        self.u = u
        return self.update_self()
    # update using self.u
    def update_self(self):
        # https://zhuanlan.zhihu.com/p/326930724
        #Rank 1:
        # x_dot = self._A@self.x + self._B*u
        # self.x+=x_dot*self.Ts
        #Rank 2:
        # u = np.array(u).reshape((-1,1))
        phi = (np.eye(*self._A.shape)+self._A*self.Ts+1/2*self._A@self._A*self.Ts**2)
        self._x = phi@self.x + self._B@self.u*self.Ts
        # Or
        # self.x = phi@self.x + 1/2*(1+phi)@self._B*u*self.Ts
        # self.timer.update()
        # self.u #update u
        return self._C@self.x + self._D@self.u
    def set_t_stop(self,t):
        self.timer.stop = t
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
    def init(self,x):
        self._x = x
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
        
class Variable():
    '''
    function warpper for varible
        y = fn(t,*fcn_args,**kwards)
    you can get y = varible.u if need
    you should use update() while you want to get a new value 
    Sample:
        u = Variable(fn=Variable.random,fcn_args = (1,2))
    Or define a function:
        def fcn(t,a,b):
            return a*t+b
    then
        u = Variable(0,1,fn=fcn,a=1,b=2)
    or:
        u = Variable(0,1,fn=fcn,fcn_args = [1,2])
    '''
    def __init__(self,t=0.0,step=0.01,stop = 10,fn = None,**kwards) -> None:
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
    def init(self):
        self.set_t(0)
    def set_t(self,t):
        self.t = t
    def __iter__(self):
        self.set_t(0)
        while True:
            yield self.y
            self.update()
            if self.t>self.stop:
                break
    @staticmethod
    def random(t,*args,**kwards):
        return np.random.randn(*args)
    def update(self):
        self.set_t(self.t+self.step)
    @property
    def y(self):
        ret = self.t
        if self.fcn ==None:
            return  self.t
        if self.fcn_args is not None:
            ret =  self.fcn(self.t,*self.fcn_args,**self.fcn_kwards)
        else:
            ret =  self.fcn(self.t,**self.fcn_kwards)
        self.update()
        return ret
class TimerVariable(Variable):
    '''
    Variable with Timer
    When you get y,it does not update,
    you should update() it
    '''
    def __init__(self,timer = Timer(),fn = None,**kwards):
        self.timer = timer
        super().__init__(timer.t,timer.step,timer.stop,fn,**kwards)
    def update(self):
        self.timer.update()
    # do not update by itself
    @property
    def t(self):
        return self.timer.t
    @t.setter
    def t(self,t):
        self.timer._t = t
    @property
    def y(self):
        ret = self.t
        if self.fcn ==None:
            return  self.t
        if self.fcn_args is not None:
            ret =  self.fcn(self.t,*self.fcn_args,**self.fcn_kwards)
        else:
            ret =  self.fcn(self.t,**self.fcn_kwards)
        return ret
class WhiteNoise(Variable):
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
            yield self.y
            if self.t>self.stop:
                break
    @property
    def y(self):
        # return TimeVariable.random(0,*self.shape)*self.sigma+self.mu
        return super().y*self.sigma+self.mu
