from state_space.base import Timer
from state_space.transfer_function import Transfer,Gain
from state_space.simulator import TimerVariable
from state_space.system_graph import Concat,SystemGraph
import numpy as np
import matplotlib.pyplot as plt
from state_space.system_graph import SystemGraph
timer = Timer()
T = 5
G = Transfer((1,),(T,1),timer=timer)
u = TimerVariable(timer,lambda x:1)
H = Gain(2)
# graph = SystemGraph([u,G,H],1,(((0,2),1,(1,-1)),(1,2)),timer)
graph = SystemGraph([u,G],1,((0,1),))
y1 = graph.simulate(False)
# G.init(0)
# graph = SystemGraph([u,G],1,(((0,1),1,(1,-1)),))
# y2 = graph.simulate(False)
plt.plot(y1[:,0],y1[:,1])
plt.show()