from worm import Worm3D
import numpy as np
import pybullet_data
import time
import pybullet as p
import pyrosim.pyrosim as ps
import matplotlib.pyplot as plt
import os

os.makedirs('plots', exist_ok=True)

speed = 1/500

def plot_signals(wormywormyworm, A, w, t, phi, title, save=False):
    fig, ax = plt.subplots()
    for s in range(wormywormyworm.num_segs):
        if s == 0:
            opacity = 1.0
        else:
            opacity = 0.2
        ax.plot(A * np.sin(w * t + s * phi), alpha=opacity)

    plt.xlabel("Time")
    plt.ylabel("Signal Amplitude")
    plt.title(title)

    if save:
        plt.savefig('plots/'+title+'.png', dpi=300)
    plt.show()



def run_single(duration):
    # Initialize the physics engine
    physicsClient = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -9.8)
    planeId = p.loadURDF("plane.urdf")
    p.changeDynamics(planeId, -1, lateralFriction=1)

    # Set parameters
    camera_distance = 30  # Distance from the target
    camera_yaw = 0  # Horizontal rotation (degrees)
    camera_pitch = -90.1  # Vertical rotation (degrees)
    camera_target_position = [0, 0, 0]  # The point the camera looks at

    # Update the camera view
    p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                 cameraYaw=camera_yaw,
                                 cameraPitch=camera_pitch,
                                 cameraTargetPosition=camera_target_position)

    x = 0
    y = 0
    z = 0.5

    length = 10
    width = 1
    segs = 10
    wormywormyworm = Worm3D(length, width, [x,y,z], segs)
    robotId = wormywormyworm.spawn()
    ps.Prepare_To_Simulate(robotId)

    stepsize = 10

    t = np.linspace(0, 10 * np.pi, duration)
    A = 0.5
    w = -np.pi/2
    phi = np.pi/5

    for i in range(duration):
        wormywormyworm.step(stepsize, (A, w, t[i], phi))
        p.stepSimulation()
        time.sleep(speed/stepsize)



    p.disconnect()
    plot_signals(wormywormyworm, A, w, t, phi, 'Single Simple Worm Signal', True)






run_single(10000)





