import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

class Motor:
    def __init__(self,pin):

        self._speed = 0
        self.setUp()

    def setUp(pin):
        '''GPIO Setup & PWM Setup'''
        GPIO.setup(pin, GPIO.OUT)
        
    @property
    def speed(self):
        return self._speed
    @speed.setter
    def speed(self,speed):
        '''software PWM 사용'''
        if speed not in range(0,101):
            raise ValueError('speed ranges fron 0 to 100, not "{0}"'.format(speed))
        '''SetUp PWM'''
        

class DCMotor(Motor):
    def __init__(self):
        pass


class ServoMotor(Motor):
    def __init__(self):
        pass


def test():
    pass