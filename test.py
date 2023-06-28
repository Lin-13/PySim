from state_space.base import Timer
from state_space.transfer_function import Transfer
import numpy as np
timer = Timer(1,0.1)
a = Transfer([1],[1,2,1],0,timer,u=0)
np.concatenate((np.array([[a._u_0.y]]),a._du))
a.u = 1
a.update(a.u)
a.update(2)
# for t in a:
#     pass
#     # print(a.y)