import math


class ParkingControl:
    def __init__(self, car, parkginLot):

        self.car = car
        self.parkingLot = parkginLot

    def StartParking(self, cur_x: float, cur_y: float) -> float:

        min_r = self.parkingLot.wall_offset + self.car.position.y  # min radius
        self.car.stop()

        target_pos_x = abs(self.car.position.x - self.env.k1.x) + min_r
        self.car.move_x(target_pos_x)  # move by x-axis

        slope = (self.car.width + min_r) / self.car.wheel_base  # calculate steering angle
        beta = math.atan(slope)

        self.car.steer(beta)
        self.car.drive(-100, self.endCondition)  # go backwards until close to the wall

    def endCondition(self):
        """주차 종료 조건"""
        if abs(self.car.position.y - self.env.space) < self.env.limit:
            return True
        else:
            return False