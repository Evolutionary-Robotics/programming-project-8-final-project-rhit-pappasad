from abc import ABC, abstractmethod
import numpy as np
from bodies.body import *


class Sensor(ABC):

    def __init__(self, parent):
        self.x = parent.x
        self.y = parent.y
        self.direction = parent.angle
        self.parent = parent

    def update(self):
        self.x = self.parent.x
        self.y = self.parent.y
        self.direction = self.parent.angle

    @abstractmethod
    def Sense(self, object):
        pass


class Auditory(Sensor):

    def __init__(self, parent, detection_radius, audio_threshold):
        super().__init__(parent)
        self.detection_radius = detection_radius
        self.threshold = audio_threshold

    def Sense(self, object):
        if not hasattr(object, '_audio'):
            print("<<<ERROR>>>  bodies -> sensors.py -> Auditory -> Sense(): object has no audio metric")
            sys.exit()

        to_ret = [0.0, self.detection_radius]
        distance = np.sqrt((self.x - object.x)**2 + (self.y - object.y)**2)
        if distance <= self.detection_radius:
            if object.audio >= self.threshold:
                to_ret[0] = 1.0
                noise = np.clip(np.random.normal(), -1.0, 1.0) * distance
                to_ret[1] = distance + noise

        return to_ret


class Visual(Sensor):

    def __init__(self, parent, vision_range, direction, fov_angle):
        super().__init__(parent)
        self.vision_range = vision_range
        self.fov = fov_angle
        self.direction = direction

    def Sense(self, object, obscuration=0.0):
        to_ret = [0.0, self.vision_range]
        distance = np.sqrt((self.x - object.x)**2 + (self.y - object.y)**2)
        if distance <= self.vision_range and abs(self.direction) < self.fov/2:
            to_ret[0] = 1.0
            noise = np.clip(np.random.normal(), -1.0, 1.0) * obscuration
            to_ret[1] = distance + noise

        return to_ret




