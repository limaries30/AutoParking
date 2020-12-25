from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)
import logging

logging.getLogger("tensorflow").disabled = True

import random

random.seed(1)
import numpy as np

np.random.seed(1)
import tensorflow as tf

tf.set_random_seed(0)

from Car import Car
import config
from env.ParkingLot import ParkingLot
from algorithms.ParkingControl import ParkingControl


if __name__ == "__main__":
    print("here")
    car = Car(config, ParkingLot, ParkingControl)
    car.start()
