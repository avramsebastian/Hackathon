import math
from entities.Directions import Directions
from entities.Role import Role

class Car:
    def __init__(self, x=0.0, y=0.0, speed=0.0, direction=Directions.FORWARD, role=Role.CIVILIAN):
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = direction
        self.role = role

    def distance_to_center(self):
        return math.hypot(self.x, self.y)