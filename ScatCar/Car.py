import math
import config
from env.ParkingLot import ParkingLot
from algorithms.ParkingControl import ParkingControl
from algorithms.LaneDetector import LaneDetector


class Car:
    def __init__(self, config, env, parkingControl):

        self.width = 10
        self.height = 30
        self.wheel_base = 20
        self.front_overhang = 5
        self.rear_overhang = 5
        self.parkingMode = parkingControl(self, env)

        self.start_pos_x = 0
        self.config = config
        self.env = env

        self.laneDetector = LaneDetector(config)
        self.curMode = None

    @property
    def position(self):
        return self.get_position()

    def get_position(self):
        """현재 위치"""
        """초음파 센서"""
        return {"x": 0, "y": 0}

    def leveling(self):
        """needs development"""
        pass

    def move_x(self, speed, target_x):
        """특정 속도로 x축 주행"""
        while self.position.x < target_x:
            # move X
            pass

    def steer(self, angle):
        """'서보 모터 각도 조절"""
        pass

    def stop(self):
        pass

    def drive(self, speed, condition):
        """특정 조건까지 일정속도로 주행"""
        while condition():
            pass

    def park(self):
        self.parkingMode.StartParking()

    def dc_ratio(self, radius: float, width: float) -> float:
        return radius / (radius + width)


def test():
    car = Car(config, ParkingLot, ParkingControl)

    car.park()


test()