import math
from ParkingLot import ParkingLot


class Car:

    def __init__(self,env,parkingControl):
        
        self.width = 10
        self.height = 30
        self.wheel_base = 20
        self.front_overhang = 5
        self.rear_overhang = 5
        self.parkingMode = parkingControl


        self.env = env
        self.curMode= None

    @property
    def position(self):
        return self.get_position()

    def get_position(self):
        '''현재 위치'''
        return [0,0]


def dc_ratio(radius:float,width:float)->float:
    return radius/(radius+width)

def calc_radius(cur_x,k1,min_r):
    return abs(cur_x-k1)+min_r

def StartParking(cur_x:float,cur_y:float)->float:

    min_r = wall_offset+cur_y

    while True:
        #slow down
        target_pos =CalcRadius(cur_x,k1,min_r)
        #move to target pos 
        dc_motor_ratio =dc_ratio(min_r,width)
        #park
        #stop


    
#     wall_offset = lot_height-margin-rear_overhang
#     back_right_radius = wall_offset+cur_y
#     front_left_raidus = math.sqrt(cur_y**2+wheel_base**2)

#     if not CheckCollision(cur_y,min_radius,target_dist): #front-left : 주차장 상단 모서리
#         #move downwards
#         return -1
#     if back_right_radius+cur_x != k1_x:  #back-wheel : 반지름 맞추기
#         #move x direction
#         return -1
    
#     radius = min_radius+width/2

#     return radius

# def CheckCollision(pos,radius,target_dist):
#     if pos+radius<=target_dist:
#         return False
#     else:
#         return True

    
