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
        self.network = None

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

    def getShape(self, shape: str):
        if shape == 'rectangle':
            top_left = (self.x - self.size[0] / 2, self.y - self.size[1] / 2)
            width = self.size[0]
            height = self.size[1]
            return (('rect', self.color, (self.x, self.y), width, height, self.angle),)  # Include angle
        elif shape == 'triangle':
            # Define the points for the triangle with the flat side aligned with the angle
            base_half_length = self.size[0] / 2
            height = self.size[1] / 2

            # Calculate triangle points with flat side perpendicular to the angle
            point1 = (self.x + base_half_length * np.cos(self.angle + np.pi / 2),
                      self.y + base_half_length * np.sin(self.angle + np.pi / 2))
            point2 = (self.x - base_half_length * np.cos(self.angle + np.pi / 2),
                      self.y - base_half_length * np.sin(self.angle + np.pi / 2))
            point3 = (self.x + height * np.cos(self.angle),
                      self.y + height * np.sin(self.angle))

            # Add more shapes as needed
            return (('polygon', self.color, (self.x, self.y), [point1, point2, point3], self.angle),)  # Include angle
        elif shape == 'circle':
            return (('circle', self.color, (self.x, self.y), self.radius),)
        # Add more shapes as needed

