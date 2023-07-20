import numpy as np
from .base import BaseBlock,Timer

class Variable(BaseBlock):
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
        #just return self.y do not change timer
        return self.y
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
