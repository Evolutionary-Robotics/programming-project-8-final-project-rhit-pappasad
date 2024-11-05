import numpy as np
from bodies.body import *

WORM_MAX_SPEED = 5.0
WORM_MAX_ACCELERATION = 1.0
WORM_MAX_FORCE = 2000.0
WORM_DETECTION_RADIUS = 250.0
CAMEL_DANGER_SPEED = 1.0
WORM_ANGULAR_VELOCITY_RANGE = (-0.1, 0.1)
WORM_ANGULAR_ACCELERATION_RANGE = (-0.01, 0.01)
WORM_HEAD_MASS_RATIO = 5/4
WORM_TAIL_MASS_RATIO = 1/2
WORM_DEFAULT_MASS = 10.0
#size
WORM_DEFAULT_SIZE = (20.0, 20.0)
WORM_SEG_SIZE = (20.0, 20.0)
WORM_TAIL_SIZE = (20.0, 20.0)
#Colors
WORM_DEFAULT_COLOR = (122, 101, 43)
WORM_PURSUIT_COLOR = (61, 50, 21)

DEF_NUM_SEGMENTS = 5

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

        def manifest(self):
            return self.getShape(self.shape)

    def __init__(self, position, direction, num_segments=DEF_NUM_SEGMENTS):
        super().__init__(position, direction, WORM_DEFAULT_SIZE, WORM_DEFAULT_COLOR)
        self.mass = WORM_DEFAULT_MASS*WORM_HEAD_MASS_RATIO + WORM_DEFAULT_MASS*num_segments + WORM_DEFAULT_MASS*WORM_TAIL_MASS_RATIO
        self.Head = self.Segment(position, direction, self.size, 0)
        self.Head.mass = WORM_DEFAULT_MASS*WORM_HEAD_MASS_RATIO
        self.segments = [self.Head]

        x, y = position
        x += WORM_SEG_SIZE[0] * np.sin(direction)
        y += WORM_SEG_SIZE[1] * np.cos(direction)
        for i in range(num_segments):
            segment = self.Segment((x, y), direction, WORM_SEG_SIZE, i+1)
            self.segments.append(segment)
            x += WORM_SEG_SIZE[0] * np.sin(direction)
            y += WORM_SEG_SIZE[1] * np.cos(direction)
        self.Tail = self.Segment((x, y), direction, WORM_TAIL_SIZE, 1+num_segments)
        self.Tail.shape = 'rectangle'
        self.Tail.mass = WORM_DEFAULT_MASS*WORM_TAIL_MASS_RATIO
        self.segments.append(self.Tail)

    def handleUpdate(self, stepsize, camels, min_cond, max_cond):
        # Get the random action for the head
        action = self.nextAction('random')
        if len(action) != self._NUM_ACT_OUTPUTS:
            print(
                f"<<<ERROR>>> bodies -> worm.py -> handleUpdate(): Action (size {len(action)}) does not meet required dimensions ({self._NUM_ACT_OUTPUTS})")
            sys.exit()

        # Update the head segment with the new force and direction
        force, direction = action
        force = np.clip(force, 0, WORM_MAX_FORCE)
        direction = direction % (2 * np.pi)
        self.Head.handleUpdate(stepsize, force, direction)

        # Ensure boundary reflection for the head
        if self.Head.x >= max_cond[0] or self.Head.x <= min_cond[0]:
            self.Head.angle = np.pi - self.Head.angle
        if self.Head.y >= max_cond[1] or self.Head.y <= min_cond[1]:
            self.Head.angle = -self.Head.angle

        # Update the position of each segment to follow the segment in front
        for i in range(1, len(self.segments)):
            prev_segment = self.segments[i - 1]
            curr_segment = self.segments[i]

            # Calculate the distance between the current segment and the previous segment
            dx = prev_segment.x - curr_segment.x
            dy = prev_segment.y - curr_segment.y
            distance = np.hypot(dx, dy)

            # If the distance is greater than the defined segment length, move the current segment closer
            if distance > WORM_SEG_SIZE[0]:
                # Calculate the angle between the current and previous segment
                angle_to_prev = np.arctan2(dy, dx)
                move_distance = distance - WORM_SEG_SIZE[0]

                # Update the current segment's position to move towards the previous segment
                curr_segment.x += move_distance * np.cos(angle_to_prev)
                curr_segment.y += move_distance * np.sin(angle_to_prev)
                curr_segment.angle = angle_to_prev

            # Ensure boundary reflection for each segment
            if curr_segment.x >= max_cond[0] or curr_segment.x <= min_cond[0]:
                curr_segment.angle = np.pi - curr_segment.angle
            if curr_segment.y >= max_cond[1] or curr_segment.y <= min_cond[1]:
                curr_segment.angle = -curr_segment.angle

    def nextAction(self, inputs):
        if inputs == 'random':
            force = np.random.uniform(0, WORM_MAX_FORCE)
            direction = np.random.uniform(0, np.pi)
            return force, direction
        else:
            return self.network.forward(inputs)

    def reset(self):
        for segment in self.segments:
            segment.reset()

    def detectCamel(self):
        return 0.0, 0.0

    def getState(self):

        return np.array([*self.detectCamel(), *self.Head.getState()])

    def manifest(self):
        return [shp.manifest() for shp in self.segments]





