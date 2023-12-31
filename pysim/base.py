# wrapper u/uk
class InputWrapper:
    def __init__(self, u) -> None:
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


class BaseBlock:
    """
    Base Block,used for type check and common interface
    u: input of block
    y: output of block
    update(u): update state
    init(x):
    """

    def __init__(self) -> None:
        pass

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
        return None

    def update(self, x=None):
        pass

    # reset system state for last init() state
    def reset(self):
        pass


class Timer:
    """
    Step time
    step: time step
    stop:time stop
    if auto = true,update t once get t by property
    """

    def __init__(self, stop=10, step=0.01, auto=False) -> None:
        self.step = step
        self.stop = stop
        self.auto = auto
        self._t = 0.0

    def init(self):
        self._t = 0.0

    def reset(self):
        self.init()

    def copy(self):
        return Timer(self.stop, self.step, self.auto)

    @property
    def done(self):
        return self._t >= self.stop

    @property
    def t(self):
        if self.auto:
            self.update()
        return self._t

    def __iter__(self):
        self.init()
        while True:
            t = self.update()
            if t >= self.stop:
                # raise StopIteration
                break
            yield t

    def update(self):
        if self._t <= self.stop:
            self._t += self.step
        return self._t
