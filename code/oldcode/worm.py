import pyrosim.pyrosim as ps
import os
import numpy as np
import time
import pybullet as p



class Worm3D:
    _num_worms = 0
    max_wl_ratio = 0.5
    size_inc_ratio = 1.25
    os.makedirs('urdfs', exist_ok=True)
    uf_low = 0.0
    uf_sideways = 10.0
    uf_high = 1.5
    max_move_force = 500

    def __init__(self, length, width, pos, segs=10):
        self.length = length
        self.width = width
        self.num_segs = segs
        self.x, self.y ,self.z = pos
        self.exists = False
        self.id = Worm3D._num_worms
        self.sim_id = None
        Worm3D._num_worms += 1
        self.path = None
        self.segment_ids = []
        self.joints = []

    def spawn(self):
        self.path = f'urdfs/worm{self.id}.urdf'
        ps.Start_URDF(self.path)
        x, y, z = self.x, self.y, self.z
        l, w, h = self.length/self.num_segs, self.width, self.width

        for i in range(self.num_segs):
            seg_name = f'Seg_{i}'

            ps.Send_Cube(name=seg_name, pos=[x, y, z], size=[l, w, h])
            self.segment_ids.append(i)

            # Scale width and height slightly as we go along
            if w < l * self.max_wl_ratio:
                w *= self.size_inc_ratio
            if h < l * self.max_wl_ratio:
                h *= self.size_inc_ratio

            # Attach a joint between each pair of segments, except the first one
            if i > 0:
                joi_name = f'Joint_{i-1}_{i}'
                parent = f'Seg_{i-1}'
                child = seg_name

                ps.Send_Joint(
                    name=joi_name,
                    parent=parent,
                    child=child,
                    type = 'revolute',
                    position = [l, y, 0],
                    axis = [0, 0, 1]
                )
                self.joints.append(joi_name.encode('utf-8'))

        ps.End()
        self.exists = True
        self.sim_id = p.loadURDF(self.path)
        return self.sim_id

    def turn(self, angle, phi):
        if not self.exists:
            print("ERROR: Not Spawned")
            return -1.0

        # Introduce a lateral (side-to-side) movement
        for j, joint in enumerate(self.joints):
            target_pos = angle + j*phi
            ps.Set_Motor_For_Joint(
                bodyIndex=self.sim_id,
                jointName=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                maxForce=self.max_move_force
            )

    def step(self, stepsize, action):
        if not self.exists:
            print("ERROR: Not Spawned")
            return -1.0

        A, w, t, phi = action

        #Introduce a lateral (side-to-side) movement
        for j, joint in enumerate(self.joints):
            target_pos = A*np.sin(w*t + j*phi)
            ps.Set_Motor_For_Joint(
                bodyIndex = self.sim_id,
                jointName = joint,
                controlMode = p.POSITION_CONTROL,
                targetPosition = target_pos,
                maxForce = self.max_move_force
            )

        for id in self.segment_ids:
            #determine motion direction
            wave_pos = w*t + id*phi
            if wave_pos < 0:
                # Moving backward, apply high friction
                p.changeDynamics(self.sim_id, id, anisotropicFriction=[self.uf_low, self.uf_sideways, self.uf_high])
            else:
                # Moving forward, apply low friction
                p.changeDynamics(self.sim_id, id, anisotropicFriction=[self.uf_high, self.uf_sideways, self.uf_low])



