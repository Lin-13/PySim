from .base import BaseBlock,Timer,InputWarpper
class PIDController(BaseBlock):
    def __init__(self,kp=1.0,ki = 0.0,kd = 0.0,u = None,timer = Timer()):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.timer = timer
        self._x = 0
        self._xi = 0
        self._xd = 0
        self._u = InputWarpper(u)
    def init(self):
        self._x,self._xd,self._xi = 0,0,0
        self.timer.init()
    @property
    def u(self):
        return self._u.y
    @u.setter
    def u(self,u):
        self._u = InputWarpper(u)
    @property
    def y(self):
        return self.kp*self._x + self.ki*self._xi+self.kd*self._xd
    def update(self,u = None):
        if u == None:
            pass
        else:
            self.u = u
        self._xd = (self.u-self._x)/self.timer.step
        self._xi += self.u*self.timer.step
        self._x = self.u
        return self.y