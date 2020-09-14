from motors import ServoMotor,DCMotor,Motor

class FrontWheels:
    
    def __init__(self,motor:Motor):
        motorLeft = motor
        motorRight= motor

class BackWheels:
    
    def __init__(self,motor:Motor):
        motorLeft = motor
        motorRight= motor
        

def test():
    frontWheels = FrontWheels(ServoMotor)
    backWheels = BackWheels(DCMotor)

    