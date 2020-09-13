class Motor:
    def __init__(self):

        self._speed = 0
        pass

    @property
    def speed(self):
        return self._speed
    @speed.setter
    def spped(self,speed):
        if speed not in range(0,101):
            raise ValueError('speed ranges fron 0 to 100, not "{0}"'.format(speed))


class DCMotor(Motor):
    def __init__(self):
        pass


class ServoMotor(Motor):
    def __init__(self):
        pass
