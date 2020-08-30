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

class parkingContorl:

    def __init__(self,car,parkginLot):
        self.car = car 
        self.parkingLot = parkginLot

    def StartParking(self,cur_x:float,cur_y:float)->float:

        min_r = self.parkingLot.wall_offset+self.car.position.y
        #stop
        target_pos_x = abs(self.car.position.x-self.env.k1.x)+min_r
        #MoveTo(target_pox_x)
        #Park

