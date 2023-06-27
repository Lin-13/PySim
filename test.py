import numpy as np
import matplotlib.pyplot as plt
# from state_space import simulator,system
from state_space.system import Pendulm
from state_space.simulator import DynamicSystem,TimeVariable,WhiteNoise

from state_space.kalman_filter import KalmanFilter
from state_space.simulator import DiscreteDynamicSystem

pendulm = Pendulm(1,0.2)
# simulator = DynamicSystem(pendulm.A,pendulm.B,pendulm.C,pendulm.D,0.001)
# simulator.init(np.array([[np.pi/10],[0]]))
# simulator.set_u(0)
# simulator.set_t_stop(10)
# history = []
# for i in simulator:
#     history.append([simulator.t,i.item()])
# history = np.array(history)
# plt.plot(history[:,0],history[:,1])
# plt.show()

simulator = DynamicSystem(pendulm.A,pendulm.B,pendulm.C,pendulm.D,0.01)
dis_simulator = DiscreteDynamicSystem(*DiscreteDynamicSystem.from_continous(simulator,np.array([[np.pi/10],[0]])))
filter = KalmanFilter(dis_simulator._A,dis_simulator._B,dis_simulator._C,dis_simulator._x,np.diag((0,1)))

simulator.init(np.array([[np.pi/10],[0]]))
filter.init(simulator.x)
simulator.set_u(0)
simulator.set_t_stop(20)
noise = WhiteNoise(0,0.3)
measure_noise = WhiteNoise(0,0.1)
simulator.u = noise
history = []
for i in simulator:
    y = i.item()+measure_noise.u.item()
    pre = filter.predict(simulator.u.u)
    filter.update(y)
    history.append([simulator.t,y,pre])
history = np.array(history)
plt.plot(history[:,0],history[:,1])
print(simulator.u)