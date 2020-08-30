import math
from ParkingLot import ParkingLot
import parkingControl from ParkingControl

class Car:

    def __init__(self,env,parkingControl):
        
        self.width = 10
        self.height = 30
        self.wheel_base = 20
        self.front_overhang = 5
        self.rear_overhang = 5
        self.parkingMode = parkingControl(self,env)


        self.env = env
        self.curMode= None

    @property
    def position(self):
        return self.get_position()

    def get_position(self):
        '''현재 위치'''
        '''초음파 센서'''
        return {'x':0,'y':0}

    def leveling(self):
        '''needs development'''
        pass

    def dc_ratio(self,radius:float,width:float)->float:
        return radius/(radius+width)