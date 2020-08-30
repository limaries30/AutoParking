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
        #ParkEnd

