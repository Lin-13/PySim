import numpy as np

# pendulum system
class Pendulm():
    def __init__(self,m,l):
        self.m = m
        self.g = 9.81
        self.l = l
        self.Ts = 0.01
    @property
    def A(self):
        return np.array([[0,1],[-self.g/self.l,0]])
    @property
    def B(self):
        return np.array([[0],[1/(self.m*self.l**2)]])
    @property
    def C(self):
        return np.array([[1,0]])
    @property
    def D(self):
        return np.array([0])
class InvertPendulm():
    '''
    (J+m*L**2)*ddtheta - mgl*theta = ml*ddx
    (M+m)*ddx - ml*ddtheta = F = u
    state space:
        X = [x,x_dot,y,y_dot].T
        y = theta = [0,0,1,0]*X
        A =[0,1,0,0;
            0,0,-m^2*g*l^2/(J*(M+m)+M*m*l^2),0;
            0,0,0,1;
            0,0,m*(M+m)*g*l/(J*(M+m)+M*m*l^2),0];
        B =[0;
            (J+m*l^2)/(J*(M+m)+M*m*l^2);
            0;
            -m*l/(J*(M+m)+M*m*l^2)];
    '''
    def __init__(self,m,M,L):
        self.g = 9.8
        self.m = m
        self.J = 1/3*m*L**2
        self.L = L
        self.M = M
    @property
    def A(self):
        m,M,g,L,J = self.m,self.M,self.g,self.L,self.J
        return np.array([[0,1,0,0],
                         [0,0,-m**2*g*L**2/(J*(M+m)+M*m*L**2),0],
                         [0,0,0,1],
                         [0,0,m*(M+m)*g*L/(J*(M+m)+M*m*L**2),0]])
    @property
    def B(self):
        m,M,g,L,J = self.m,self.M,self.g,self.L,self.J
        return np.array([[0],
                         [(J+m*L**2)/(J*(M+m)+M*m*L**2)],
                         [0],
                         [-m*L/(J*(M+m)+M*m*L**2)]])
    @property
    def C(self):
        return np.array([[0,0,1,0]])
    @property
    def D(self):
        return np.array([[0]])