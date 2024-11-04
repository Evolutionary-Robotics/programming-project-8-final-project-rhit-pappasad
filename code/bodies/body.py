import numpy as np
from abc import ABC, abstractmethod
import sys

class Body(ABC):
    _VALIDSHAPES = {'circle', 'x', 'rectangle', 'triangle'}

    def __init__(self, position, direction, size, color):
        self.x, self.y = position
        self.speed = 0.0
        self.linear_acceleration = 0.0

        self.angle = direction
        self.angular_velocity = 0.0
        self.angular_acceleration = 0.0

        self.size = size
        self.radius = np.sqrt(self.size[0] ** 2 + self.size[1] ** 2)
        self._shape = None
        self.color = color
        self.sensors = []

    def reset(self):
        self.angular_velocity = 0.0
        self.speed = 0.0
        self.linear_acceleration = 0.0
        self.is_alive = True

    def step(self, stepsize):
        self.x += stepsize * self.speed * np.cos(self.angle)
        self.y += stepsize * self.speed * np.sin(self.angle)
        self.speed += stepsize * self.linear_acceleration

        self.angle += stepsize * self.angular_velocity
        self.angular_velocity += stepsize * self.angular_acceleration

        while self.angle > np.pi*2:
            self.angle -= np.pi*2
        while self.angle < 0:
            self.angle += np.pi*2

    @abstractmethod
    def handleUpdate(self, stepsize, *args):
        pass

    @abstractmethod
    def getState(self):
        pass

    @abstractmethod
    def manifest(self):
        pass

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            print(f"<<<ERROR>>>   bodies -> Body.py -> shape(): Body {self.__class__.__name__} has no shape")
            sys.exit(2)

        return self._shape

    @shape.setter
    def shape(self, shape: str):
        if shape not in self._VALIDSHAPES:
            print(f"<<<ERROR>>>   bodies -> Body.py -> shape setter(): {shape} is not a valid shape")
            sys.exit(1)

        shape = shape.lower()
        if shape == 'circle':
            shape = (self.color, (self.x, self.y), self.radius)
        elif shape == 'x':
            thickness = int(self.size[0] / 3)
            s1 = (self.x - (self.size[0] / 2), self.y - (self.size[1] / 2))  # upper left
            e1 = (self.x + (self.size[0] / 2), self.y + (self.size[1] / 2))  # lower right
            s2 = (self.x - (self.size[0] / 2), self.y + (self.size[1] / 2))  # lower left
            e2 = (self.x + (self.size[0] / 2), self.y - (self.size[1] / 2))  # upper right
            shape = ((255, 0, 0), s1, e1, thickness), ((255, 0, 0), s2, e2, thickness)
        elif shape == 'rectangle':
            #draw square
            pass
        elif shape == 'triangle':
            #draw tri
            pass
        self._shape = shape
