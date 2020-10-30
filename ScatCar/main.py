from Car import Car
import config
from env.ParkingLot import ParkingLot
from algorithms.ParkingControl import ParkingControl



if __name__ =="__main__":
    
    car = Car(config,ParkingLot,ParkingControl)
    car.start()
