import matplotlib.pyplot as plt
from .base import Timer, BaseBlock
import numpy as np


class Concat(BaseBlock):
    """
    Concat SUM of systems, like:\n
    Concat.u = system[0].y + system[1].y …\n
    @params\n
        system:concat system, must be a list.
        Each system must be initialized.\n
        sign:tuple or int,indicate a system to be added or sub.* ADD DEFAULT *\n
    like:\n
    con = Concat([system1,system2],(1,-1))\n
    => con.u = system1.y - system.y\n
        con.y = con.u
    """

    def __init__(self, system: list, sign=None):
        super().__init__()
        self.system = []
        self.set_system(system)
        self.sign = ()
        self.set_sign(sign)

    def set_system(self, system):
        assert (isinstance(system, list))
        # Check whether the system has system.y
        for i in system:
            try:
                i.__getattribute__("y")
            except AttributeError:
                raise AttributeError("Concat system must has attribute y as output")
            # if not isinstance(i,BaseBlock):
            #     raise TypeError("System must be subclass of BaseBlock")
        self.system = system

    def set_sign(self, sign):
        if not (isinstance(sign, tuple) or isinstance(sign, list)):
            if not isinstance(sign, int):
                raise TypeError
            sign = (sign,) * len(self.system)
        else:
            sign = tuple(sign)
            if len(sign) < len(self.system):
                sign = sign + (1,) * (len(self.system) - len(sign))
            else:
                sign = sign[0:len(self.system)]
        self.sign = sign

    @property
    def u(self):
        sum = 0
        for i in range(len(self.system)):
            sum += self.sign[i] * self.system[i].y
        return sum

    @property
    def y(self):
        return np.array(self.u)

    def update(self, u=None):
        return self.y


class SystemGraph(BaseBlock):
    """
    @Params:
        systems: list of System Block,which block should be subclass of BaseBock
                SystemMap would use attribute:u,y,update,timer[option,if it needs]
        connections:tuple of connection. And each element must be another tuple of int\n
            [or tuple,means Concat of system block,see Example]\n
            if connection is(tuple,int,tuple)\n
            ((index of system),index of concat,sign of concat[tuple])\n
            if index of concat is a Concat,itself will be used,\n
            otherwise class will create one(save in self.concats) and connect system[tuple] & system[int]\n
    Example:\n
        SystemGraph([s0,s1,s2],(((0,2),1,(1,-1)),(1,2))\n
        (1,2) means s2.u = s1\n
        ((0,2),1,(1,-1)) means s2.u = Concat([s0,s2],(1,-1))\n
    """

    def __init__(self, systems: list[BaseBlock], output, connections: tuple, timer=Timer()):
        super().__init__()
        self.systems = systems
        self.timer = timer
        self.connections = connections
        self.output = output
        self.concats = []

        self.set_timer(self.timer)
        self.connect(connections)

    def set_timer(self, timer):
        # set timer for each system if it needs timer
        for block in self.systems:
            try:
                block.__getattribute__("timer")
                block.__setattr__("timer", timer)
            except AttributeError:
                pass
        self.timer = timer

    def connect(self, connections: tuple):
        self.connections = connections
        for i, connect in enumerate(self.connections):
            assert (isinstance(connect, tuple))
            if len(connect) == 2:
                self.systems[connect[1]].u = self.systems[connect[0]]

            elif len(connect) == 3:
                # if connection is(tuple,int,tuple)
                # ((index of a system), index of concat, sign of concat[tuple])
                if isinstance(connect[0], tuple) and isinstance(connect[2], tuple):
                    if self.systems[connect[1]] is Concat:
                        self.systems[connect[1]].set_system([self.systems[i] for i in connect[0]])
                        self.systems[connect[1]].set_sign(tuple(connect[2]))
                    else:  # if connect[1] not Concat, create one to connect connect[0] and connect[1]
                        # print("Create a new Concat Block")
                        self.concats.append(Concat([self.systems[i] for i in connect[0]], connect[2]))
                        self.systems[connect[1]].u = self.concats[-1]
                        # print(self.systems[connect[1]],self.systems[connect[1]]._u._u)
            else:
                raise TypeError(f"input (int,int)or (tuple,int,tuple) for connections\n\
                                While function get {len(connect)} element in connection {i}")

    def __repr__(self) -> str:
        string = "\n".join(
            [f"System {i}:{system.__class__.__name__}\t id:{id(system)}" for (i, system) in enumerate(self.systems)])
        return string

    def __iter__(self):
        for t in self.timer:
            self.update()
            yield self.systems[self.output].y

    def reset(self):
        for system in self.systems:
            system.reset()
        self.timer.reset()

    def simulate(self, plot=True):
        h = []
        for y in self:
            h.append([self.t, y.item()])
        h = np.array(h)
        if plot:
            plt.plot(h[:, 0], h[:, 1])
            plt.show()
        return h.copy()

    @property
    def t(self):
        return self.timer.t

    def init(self, x=None):
        pass

    @property
    def u(self):
        return None

    @u.setter
    def u(self, u):
        pass

    @property
    def y(self):
        return self.systems[self.output].y

    def update(self, x=None):
        for s in self.systems:
            s.update()
