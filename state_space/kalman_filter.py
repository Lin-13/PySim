import numpy as np
from .simulator import WhiteNoise
class KalmanFilter():
    """
    KalmanFilter:
        x_k = A_k * x_k-1 + (B*) u_k +w_k
        z_k = C_k * x_k + v_k
        z_k and u_k must be 1-dim
    st:
        w_k ~ N(0,R),v_k ~ N(0,Q)
    """
    def __init__(self,A0,B0,C0,x0,P0,R = 0,Q =  1e-3):
        self._A = A0
        self._B = B0
        self._C = C0
        self._k = 0
        self._x = x0
        self.R = R
        self.Q = Q
        self._w = WhiteNoise(0,self.R,shape=x0.shape)
        self._v = WhiteNoise(0,self.Q)
        self._P = P0
        # self._temp_state = {}
        self.x_hat = self._x
        self.P_hat = self._P
    @property
    def A(self):
        A = self._A
        # try:
        #     A = self._A.__getattribute__("u")
        #     return self._A
        # except AttributeError:
        #     A = self._A.u
        #     return self._A.u
        # else:
        return A
    @property
    def C(self):
        # try:
        #     self._C.__getattr__("u")
        #     return self._C
        # except AttributeError:
        #     return self._C.u
        return self._C
    @property
    def x(self):
        return self._x
    @property
    def P(self):
        return self._P
    @property
    def w(self):
        return self._w.y
    @property 
    def v(self):
        return self._v.y
    def init(self,x):
        self._x = x
        self._k = 0
    def predict(self,uk):
        x_hat = self.A@self.x + self._B*uk
        P_hat = self.A@self.P@self.A.T + self.R
        self.x_hat= x_hat
        self.P_hat = P_hat
        self._k+=1
        return x_hat,self.C@x_hat
    def update(self,z):
        x_hat = self.x_hat
        P_hat = self.P_hat
        # K = P_hat@self.C.T@np.linalg.inv(self.C@P_hat@self.C.T+self.Q)
        K = P_hat@self.C.T/(self.C@P_hat@self.C.T+self.Q)
        self._x = K@(z - self.C@x_hat)
        I = np.eye(*P_hat.shape)
        self._P = (I - K@self.C)@P_hat
        