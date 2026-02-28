import math


class Car:
	def __init__(self, x=0.0, y=0.0, direction=0.0, speed=0.0):
		self.x = x
		self.y = y
		self.direction = direction
		self.speed = speed
		
	def distance_to_center(self):
		return math.hypot(self.x, self.y)
