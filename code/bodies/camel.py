import numpy as np
from bodies.body import *
from bodies.sensors import *

CAMEL_MAX_SPEED = 2.0
CAMEL_ACCELERATION_RANGE = (-0.5, 0.5)
CAMEL_MAX_FORCE = 150.0
CAMEL_HEARING_RADIUS = 50.0
CAMEL_VISION_RANGE = 100.0
CAMEL_FOV = 3*np.pi/4
WORM_HEARING_SPEED = 3.5
CAMEL_ANGULAR_VELOCITY_RANGE = (-np.pi/8, np.pi/8)
CAMEL_ANGULAR_ACCELERATION_RANGE = (-0.05, 0.05)
CAMEL_DEFAULT_MASS = 1.0
#size
CAMEL_DEFAULT_SIZE = (5.0, 5.0)
#Colors
CAMEL_DEFAULT_COLOR = (255, 0, 0)

class Camel(Body):
    _NUM_ACT_OUTPUTS = 2

    def __init__(self, position, direction):
        super().__init__(position, direction, CAMEL_DEFAULT_SIZE, CAMEL_DEFAULT_COLOR)
        self.mass = CAMEL_DEFAULT_MASS
        self.Ears = Auditory(self, CAMEL_HEARING_RADIUS, WORM_HEARING_SPEED)
        self.Eyes = Visual(self, CAMEL_VISION_RANGE, direction, CAMEL_FOV)
        self._audio = self.speed
        self.shape = 'circle'

    def handleUpdate(self, stepsize, worms, pools, min_cond, max_cond):
        # Get the action for the head
        if len(worms) > 0:
            inputs = self.getState(worms, pools)
            action = self.nextAction(inputs)
        else:
            action = self.nextAction('random')

        if len(action) != self._NUM_ACT_OUTPUTS:
            print(
                f"<<<ERROR>>> bodies -> worm.py -> handleUpdate(): Action (size {len(action)}) does not meet required dimensions ({self._NUM_ACT_OUTPUTS})")
            sys.exit()

        force, direction = action
        force = np.clip(force, -CAMEL_MAX_FORCE, CAMEL_MAX_FORCE)
        direction = direction % (2 * np.pi)
        self.linear_acceleration = np.clip(force * self.mass, *CAMEL_ACCELERATION_RANGE)
        torque = self.radius * force * np.sin(direction)
        moment_of_inertia = self.mass * self.radius**2
        self.angular_acceleration = np.clip(torque / moment_of_inertia, *CAMEL_ANGULAR_ACCELERATION_RANGE)


        # Ensure boundary reflection for the head
        if self.x >= max_cond[0] or self.x <= min_cond[0]:
            self.angle = np.pi - self.angle
        if self.y >= max_cond[1] or self.y <= min_cond[1]:
            self.angle = -self.angle

        # Move the sensor position
        self.Ears.update()
        self.Eyes.update()

        self.step(stepsize)

        # Clip speed and angular velocity within allowed limits
        self.speed = np.clip(self.speed, 0, CAMEL_MAX_SPEED)
        self.angular_velocity = np.clip(self.angular_velocity, *CAMEL_ANGULAR_VELOCITY_RANGE)

    def nextAction(self, inputs):
        if inputs == 'random':
            force = np.random.uniform(-CAMEL_MAX_FORCE, CAMEL_MAX_FORCE)
            direction = np.random.uniform(-np.pi, np.pi)
            return force, direction
        else:
            return self.network.forward(inputs)

    def detectWorms(self, worms):
        if len(worms) == 0:
            return [0.0, 0.0]
        detections = np.array([self.Ears.Sense(worm) for worm in worms])
        print(detections)
        return detections[np.argmin(detections[:, 1])]

    def detectPools(self, pools):
        if len(pools) == 0:
            return [0.0, 0.0]
        detections = np.array([self.Eyes.Sense(pool) for pool in pools])
        return detections[np.argmin(detections[:, 1])]

    def getState(self, worms, pools):
        return np.array([*self.detectWorms(worms), *self.detectPools(pools), self.speed, self.angle, self.angular_velocity, self.angular_acceleration, self.linear_acceleration])

    def manifest(self):
        return self.getShape(self.shape)





