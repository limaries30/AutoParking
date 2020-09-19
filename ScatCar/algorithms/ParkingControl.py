import math

class ParkingControl:
    
    def __init__(self,car,parkginLot):
        
        self.car = car 
        self.parkingLot = parkginLot

    def StartParking(self,cur_x:float,cur_y:float)->float:

        min_r = self.parkingLot.wall_offset+self.car.position.y #최소 반지름
        self.car.stop()

        target_pos_x = abs(self.car.position.x-self.env.k1.x)+min_r
        self.car.move_x(target_pos_x)  #최소반지름 위치로 x축 이동

        slope = (self.car.width+min_r)/self.car.wheel_base #회전각 계산
        beta = math.atan(slope)

        self.car.steer(beta)
        self.car.drive(-100,self.endCondition) #주차될때까지 일정속도로 후진

    def endCondition(self):
        '''주차 종료 조건'''
        if  abs(self.car.position.y-self.env.space)<self.env.limit:
            return True
        else:
            return False