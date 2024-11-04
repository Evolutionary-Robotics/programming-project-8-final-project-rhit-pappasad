from body import *

WORM_MAX_SPEED = 5.0
WORM_MAX_ACCELERATION = 1.0
WORM_MAX_FORCE = 2000.0
WORM_DETECTION_RADIUS = 250.0
CAMEL_DANGER_SPEED = 1.0
WORM_ANGULAR_VELOCITY_RANGE = (-0.1, 0.1)
WORM_ANGULAR_ACCELERATION_RANGE = (-0.01, 0.01)
WORM_HEAD_MASS_RATIO = 5/4
WORM_TAIL_MASS_RATIO = 1/2
WORM_DEFAULT_MASS = 100.0
#size
WORM_DEFAULT_SIZE = (20.0, 20.0)
WORM_SEG_SIZE = (20.0, 10.0)
WORM_TAIL_SIZE = (20.0, 20.0)
#Colors
WORM_DEFAULT_COLOR = (122, 101, 43)
WORM_PURSUIT_COLOR = (61, 50, 21)

DEF_NUM_SEGMENTS = 10

class Worm(Body):
    _NUM_ACT_OUTPUTS = 2

    class Segment(Body):
        _OSCILLATION_FREQ = np.pi / 6
        _PHASE_OFFSET = np.pi/4
        _TORQUE_SCALE_FACTOR = 0.1
        _DAMPING_FACTOR = 0.01
        _ALIGNMENT_TORQUE_SCALE_FACTOR = 0.05

        def __init__(self, position, direction, size, id):
            super().__init__(position, direction, size, WORM_DEFAULT_COLOR)
            self.shape = 'rectangle'
            self.id = id
            self.mass = WORM_DEFAULT_MASS

        def handleUpdate(self, stepsize, force, direction):
            # Existing linear acceleration update
            self.linear_acceleration = force * self.mass
            self.linear_acceleration = np.clip(self.linear_acceleration, 0, WORM_MAX_ACCELERATION)

            angle_diff = (direction - self.angle) % (2*np.pi)
            if angle_diff > np.pi:
                angle_diff -= 2*np.pi

            phase_offset = self.id * self._PHASE_OFFSET
            damping = self._DAMPING_FACTOR * self.angular_velocity

            # Simple angular acceleration based on directional change or torque
            torque = self._TORQUE_SCALE_FACTOR * np.sin(stepsize*self._OSCILLATION_FREQ + phase_offset) + self._ALIGNMENT_TORQUE_SCALE_FACTOR*angle_diff
            moment_of_inertia = self.mass * (self.size[0] ** 2 + self.size[1] ** 2) / 12  # Moment of inertia for a rectangle

            self.angular_acceleration = (torque - damping) / moment_of_inertia
            self.angular_acceleration = np.clip(self.angular_acceleration, *WORM_ANGULAR_ACCELERATION_RANGE)

            # Step function handles position and velocity updates
            self.step(stepsize)

            # Clip speed and angular velocity within allowed limits
            self.speed = np.clip(self.speed, 0, WORM_MAX_SPEED)
            self.angular_velocity = np.clip(self.angular_velocity, *WORM_ANGULAR_VELOCITY_RANGE)

        def getState(self):
            return np.array([
                self.speed,
                self.angle,
                self.angular_velocity,
                self.angular_acceleration,
                self.linear_acceleration
            ])


    def __init__(self, position, direction, num_segments=DEF_NUM_SEGMENTS):
        super().__init__(position, direction, WORM_DEFAULT_SIZE, WORM_DEFAULT_COLOR)
        self.mass = WORM_DEFAULT_MASS*WORM_HEAD_MASS_RATIO + WORM_DEFAULT_MASS*num_segments + WORM_DEFAULT_MASS*WORM_TAIL_MASS_RATIO
        self.Head = self.Segment(position, direction, self.size, 0)
        self.Head.mass = WORM_DEFAULT_MASS*WORM_HEAD_MASS_RATIO
        self.segments = [self.Head]

        x, y = position
        x += WORM_SEG_SIZE[0] * np.cos(direction)
        y += WORM_SEG_SIZE[1] * np.sin(direction)
        for i in range(num_segments):
            segment = self.Segment(x, y, WORM_SEG_SIZE, i+1)
            self.segments.append(segment)
            x += WORM_SEG_SIZE[0] * np.cos(direction)
            y += WORM_SEG_SIZE[1] * np.sin(direction)
        self.Tail = self.Segment(x, y, WORM_TAIL_SIZE, 1+num_segments)
        self.Tail.shape = 'triangle'
        self.Tail.mass = WORM_DEFAULT_MASS*WORM_TAIL_MASS_RATIO
        self.segments.append(self.Tail)

    def handleUpdate(self, stepsize, action):
        if len(action) != self._NUM_ACT_OUTPUTS:
            print(f"<<<ERROR>>> bodies -> worm.py -> handleUpdate(): Action (size {len(action)} does not meet required dimensions ({self._NUM_ACT_OUTPUTS})")
            sys.exit()

        force, direction = action
        force = np.clip(force, 0, WORM_MAX_FORCE)
        while direction > np.pi * 2:
            direction -= np.pi * 2
        while direction < 0:
            direction += np.pi * 2

        for segment in self.segments:
            segment.handleUpdate(stepsize, force, direction)

    def reset(self):
        for segment in self.segments:
            segment.reset()

    def detectCamel(self):
        return 0.0, 0.0

    def getState(self):

        return np.array([*self.detectCamel(), *self.Head.getState()])





