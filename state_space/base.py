# warpper u/uk
class InputWarpper():
    def __init__(self,u) -> None:
        self._u = u
        try:
            u.__getattribute__("y")
            self._has_warp = True
        except AttributeError:
            self._has_warp = False
    @property
    def y(self):
        if self._has_warp:
            return self._u.y
        else:
            return self._u
class BaseBlock():
    '''
    Base Block,used for type check and common interface
    u: input of block
    y: output of block
    update(u): update state
    init(x):
    '''
    def __init__(self) -> None:
        pass
    def init(self,x):
        pass
    @property
    def u(self):
        pass
    @u.setter
    def u(self,u):
        pass
    @property
    def y(self):
        pass
    def update(self,x):
        pass
class Timer():
    '''
    Step time
    step: time step
    stop:time stop
    if mode = auto,update t while get t by property
    '''
    def __init__(self,stop = 10,step = 0.01,mode = "auto") -> None:
        self.step = step
        self.stop = stop
        self.mode = mode
        self._t = 0.0
    def init(self):
        self._t = 0.0
    def copy(self):
        return Timer(self.stop,self.step,self.mode)
    @property
    def done(self):
        return self._t>=self.stop
    @property
    def t(self):
        if self.mode == "mode":
            self.update()
        return self._t
    def __iter__(self):
        self.init()
        while True:
            t =  self.update()
            if t>=self.stop:
                # raise StopIteration
                break
            yield t
    def update(self):
        if self._t<=self.stop:
            self._t+=self.step
        return self._t