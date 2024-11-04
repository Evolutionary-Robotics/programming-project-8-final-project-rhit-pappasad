import numpy as np
from abc import ABC, abstractmethod

class Body(ABC):

    def __init__(self, x, y, direction, desert, size, color):
        self.x = x
        self.y = y
        self.angle = direction
        self.angular_velocity = 0.0
        self.speed = 0.0
        self.acceleration = 0.0
        self.desert = desert
        self.desert.addAgent(self)
        self.size = size
        self.radius = np.sqrt(self.size[0]**2 + self.size[1]**2)
        self.color = color
        self.is_alive = True

    def reset(self):
        self.angular_velocity = 0.0
        self.speed = 0.0
        self.acceleration = 0.0
        self.is_alive = True

    def update(self, stepsize):
        # Update position based on angle and speed
        self.x += stepsize * self.speed * np.cos(self.angle)
        self.y += stepsize * self.speed * np.sin(self.angle)

        # Update angle with angular velocity
        self.angle += stepsize * self.angular_velocity

        # Update speed based on acceleration
        self.speed += stepsize * self.acceleration

        # Clip position to stay within desert boundaries
        self.x = np.clip(self.x, self.desert.min_x, self.desert.max_x)
        self.y = np.clip(self.y, self.desert.min_y, self.desert.max_y)

        # Handle boundary collision - reflect the angle when hitting desert edges
        if self.x >= self.desert.max_x or self.x <= self.desert.min_x:
            self.angle = np.pi - self.angle  # Reflect horizontally
        if self.y >= self.desert.max_y or self.y <= self.desert.min_y:
            self.angle = -self.angle  # Reflect vertically


    @abstractmethod
    def handleUpdate(self, stepsize, action):
        pass

    @abstractmethod
    def state(self):
        pass

    def getShape(self):
        if self.is_alive:
            return 'Circle', self.color, (self.x, self.y), self.radius
        else:
            thickness = int(self.size[0]/3)
            s1 = (self.x - (self.size[0] / 2), self.y - (self.size[1] / 2)) #upper left
            e1 = (self.x + (self.size[0] / 2), self.y + (self.size[1] / 2)) #lower right
            s2 = (self.x - (self.size[0] / 2), self.y + (self.size[1] / 2)) #lower left
            e2 = (self.x + (self.size[0] / 2), self.y - (self.size[1] / 2)) #upper right
            return 'X', [(255, 0, 0), s1, e1, thickness], [(255, 0, 0), s2, e2, thickness]

    def inRadius(self, other, radius) -> bool:
        return self.getDistance(other.x, other.y) <= radius

    def getDistance(self, x, y):
        return (np.sqrt((self.x - x)**2 + (self.y - y)**2))

    def getDirectionTo(self, x, y):
        dx = x - self.x
        dy = y - self.y
        angle = np.arctan2(dy, dx)
        angle_dif = np.arctan2(np.sin(angle - self.angle), np.cos(angle - self.angle))
        return angle_dif

    def isIntersecting(self, other) -> bool:
        return (self.x < other.x + other.size[0]
                and self.x + self.size[0] > other.x
                and self.y < other.y + other.size[1]
                and self.y + self.size[1] > other.y)


class Camel(Body):
    max_hydration = 1000.0
    max_speed = 2.0
    max_accel = 1.0
    def_size = (5.0, 5.0)
    color = (3, 252, 65)
    success_color = (0, 255, 0)
    ang_velocity_range = 0.2
    state_returns = 6
    trainables = 2
    thirsty_threshold = max_hydration/2
    fov = np.pi/2
    arrow_color = (100, 0, 100)

    def __init__(self, x, y, direction, desert, speed=0.5, size=None):
        if size is None:
            size = self.def_size
        super().__init__(x, y, direction, desert, np.array(size), self.color)
        self.hydration = self.max_hydration
        self.is_alive = True
        self.speed = speed
        self.reached_water = False
        self.max_state = np.array([self.desert.max_x, self.desert.max_y, 2*np.pi, self.max_speed, 1.0, self.desert.max_distance])

    def findPool(self) -> (float, float):
        if self.desert.pool is None:
            return 0.0, self.desert.max_distance

        direction = self.getDirectionTo(self.desert.pool.x, self.desert.pool.y)
        distance = self.getDistance(self.desert.pool.x, self.desert.pool.y)

        if abs(direction) <= self.fov / 2:
            return 1.0, distance
        else:
            return 0.0, self.desert.max_distance

    def state(self):
        return np.array([self.x, self.y, self.angle, self.speed, *self.findPool()]) / self.max_state

    def handleUpdate(self, stepsize, action):
        if not self.is_alive:
            return 0.0

        ret = self.step(stepsize, action)
        self.speed = np.clip(self.speed, 0, self.max_speed)
        return ret

    def die(self):
        self.is_alive = False

    def step(self, stepsize, action):
        #Extract relevant actions
        self.acceleration = np.clip(action[0], -self.max_accel, self.max_accel)
        self.angular_velocity = np.clip(action[1], -self.ang_velocity_range, self.ang_velocity_range)
        self.update(stepsize)

        if self.isIntersecting(self.desert.pool):
            self.reached_water = True
            self.color = self.success_color
            return stepsize
        else:
            distance_to_pool = self.getDistance(self.desert.pool.x, self.desert.pool.y) - self.desert.pool.radius
            reward = stepsize * (1 - distance_to_pool/self.desert.max_distance)
            return max(0.0, reward)

    def getArrow(self):
        thickness = 3
        mag = 20
        arrow_head_size = 0.25 * mag
        arrow_head_angle = np.pi / 6
        start = (self.x + np.cos(self.angle)*(self.size[0] / 2), self.y + np.sin(self.angle)*(self.size[1] / 2))
        end = (start[0] + mag*np.cos(self.angle), start[1] + mag*np.sin(self.angle))
        left_wing = (end[0] - arrow_head_size * np.cos(self.angle - arrow_head_angle),
                     end[1] - arrow_head_size * np.sin(self.angle - arrow_head_angle))
        right_wing = (end[0] - arrow_head_size * np.cos(self.angle + arrow_head_angle),
                      end[1] - arrow_head_size * np.sin(self.angle + arrow_head_angle))

        return 'Arrow', self.arrow_color, [start, end, thickness], [end, left_wing, thickness], [end, right_wing, thickness]

class Sandworm(Body):
    def_speed = 5.0
    detection_radius = 250.0
    speed_detection_threshold = 1.0
    ang_velocity_range = 0.1
    def_size = (20.0, 20.0)
    def_color = (122, 101, 43)
    pursuit_color = (61, 50, 21)
    pursue_time = 10

    def __init__(self, x, y, direction, desert, size=None):
        if size is None:
            size = self.def_size
        super().__init__(x, y, direction, desert, np.array(size), self.def_color)
        self.speed = self.def_speed
        self.angular_velocity = np.clip(self.angular_velocity, -self.ang_velocity_range, self.ang_velocity_range)
        self.state = 'Roaming'
        self.pursuit_clock = self.pursue_time

    def state(self):
        return np.array([self.x, self.y, self.angle, self.speed, self.angular_velocity])

    # Roaming behavior (random movement)
    def Roaming(self, stepsize, *nonargs):
        # Random angular velocity within range
        self.angular_velocity = np.random.uniform(
            -self.ang_velocity_range,
            self.ang_velocity_range
        )

        self.color = self.def_color

        self.update(stepsize)

    # Pursuit behavior (chase the prey)
    def Pursuit(self, stepsize, prey):
        angle2prey = self.getDirectionTo(prey.x, prey.y)

        # Scale angular velocity based on angle difference
        if abs(angle2prey) > 0.1:  # If not facing the prey, adjust
            self.angular_velocity = np.clip(angle2prey, -self.ang_velocity_range, self.ang_velocity_range)
        else:
            self.angular_velocity = 0  # Stop turning if almost aligned

        self.color = self.pursuit_color
        # Update position and angle
        self.update(stepsize)

    def handleUpdate(self, stepsize, prey: Camel):
        # Call the appropriate method based on current state
        self.__getattribute__(self.state)(stepsize, prey)

        if prey.inRadius(self, self.radius):
            prey.die()

        # Update state: switch to Pursuit if prey detected, else Roaming
        if self.findPrey(prey):
            self.state = 'Pursuit'
            self.pursuit_clock = self.pursue_time
        elif self.state == 'Pursuit' and self.pursuit_clock > 0:
            self.state = 'Pursuit'
            self.pursuit_clock -= 1
        else:
            self.state = 'Roaming'
        return 1.0

    #If prey in radius and is causing surface disruptions,
    #returns the coordinates of the prey, returns false otherwise
    def findPrey(self, prey):
        if self.inRadius(prey, self.detection_radius):
            if prey.speed >= self.speed_detection_threshold:
                return prey.x, prey.y
            else:
                return False
        else:
            return False






