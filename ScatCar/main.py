from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
import logging
logging.getLogger('tensorflow').disabled = True
from Car import Car
import config
from env.ParkingLot import ParkingLot
from algorithms.ParkingControl import ParkingControl



if __name__ =="__main__":
    print('here')
    car = Car(config,ParkingLot,ParkingControl)
    car.start()
