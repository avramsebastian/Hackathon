#!/usr/bin/env python3

from dataclasses import dataclass, asdict
from typing import List

# Directions can be extended as needed
DIRECTIONS = ("FORWARD", "LEFT", "RIGHT", "BACKWARD")

@dataclass
class Car:
    x: float
    y: float
    speed: float
    direction: str

    def move(self, dt: float = 1.0):
        """Move car in its current direction by speed * dt."""
        if self.direction == "FORWARD":
            self.y += self.speed * dt
        elif self.direction == "BACKWARD":
            self.y -= self.speed * dt
        elif self.direction == "LEFT":
            self.x -= self.speed * dt
        elif self.direction == "RIGHT":
            self.x += self.speed * dt

    def as_dict(self):
        return asdict(self)

class World:
    def __init__(self):
        # Initialize player car
        self.my_car = Car(x=0.0, y=0.0, speed=0.0, direction="FORWARD")

        # Initialize some traffic cars
        self.traffic: List[Car] = [
            Car(x=10.0, y=5.0, speed=5.0, direction="LEFT"),
            Car(x=-5.0, y=20.0, speed=10.0, direction="FORWARD"),
        ]

        # Current traffic sign at intersection
        self.current_sign = "YIELD"

    def update_physics(self, dt: float = 1.0):
        """Update positions of all cars."""
        self.my_car.move(dt)
        for car in self.traffic:
            car.move(dt)

    def get_ml_input(self):
        """Return the dict expected by the ML model."""
        return {
            "my_car": self.my_car.as_dict(),
            "sign": self.current_sign,
            "traffic": [car.as_dict() for car in self.traffic],
        }

# Example usage:
if __name__ == "__main__":
    world = World()
    world.update_physics()
    print(world.get_ml_input())
