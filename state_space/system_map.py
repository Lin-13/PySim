from .base import InputWarpper,Timer,BaseBlock
from .simulator import TimerVariable,Variable,DynamicSystem,DiscreteDynamicSystem
from .pid_controller import PIDController
from .transfer_function import Transfer
class Concat(BaseBlock):
    '''
    Concat SUM of systems,like:\n
    Concat.u = system[0].y + system[1].y .....\n
    @params\n
        system:concat system,must be a list.each system must be initializes.\n
        sigm:tuple or int,indicate system to be added or sub.* ADD DEFAULT *\n
    like:\n
    con = Concat([system1,system2],(1,-1))\n
    =>  con.u = system1.y - system.y\n
        con.y = con.u
    '''
    def __init__(self,system:list,sign = None):
        assert(isinstance(system,list))
        for i in system:
            if not isinstance(i,BaseBlock):
                raise TypeError("System must be subclass of BaseBlock")
        self.system = system
        if not (isinstance(sign,tuple) or isinstance(sign,list)):
            if not isinstance(sign,int):
                raise TypeError
            sign = (sign,)*len(system)
        else:
            sign = tuple(sign)
            if len(sign)<len(system):
                sign = sign + (1,)*(len(system)-len(sign))
            else:
                sign = sign[0:len(system)]
        self.sign = sign
    @property
    def u(self):
        sum = 0
        for i in range(len(self.system)):
            sum += self.sign[i]*self.system[i].y
        return sum
    @property
    def y(self):
        return self.u
    def update(self,u=None):
        return self.y
class SystemMap(BaseBlock):
    '''
    co-simulation with different system blocks\n
    each system block must be Subclass of BaseBlock\n
    this class will manage different timer.\n
    and you'd better allocation different timer in the same attribute or use a global timer.
    and you can use another timer(should be less timer step)
    
    Also,this system has INPUT u and OUTPUT y,which must be attributes of systems
    '''
    def __init__(self,systems,timers,timer = Timer(step=0.001),u = None):
        pass