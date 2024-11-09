import os
from environment import desert
from bodies import worm, camel
import pygame
import pandas as pd
import pickle
from copy import deepcopy as copy
import matplotlib.pyplot as plt
import sys
import imageio
import numpy as np




class Simulation:
    history_labels = ['camel x', 'camel y', 'camel angle', 'camel speed', 'camel angular_velocity',
                      'camel acceleration', 'camel hydration',
                      'distance']

    def __init__(self, desert: desert.Desert, bg=desert.DEF_DESERT_COLOR, noise=0.01):
        self.desert = desert
        self.bg = bg
        self.noise = noise
        self.running = False
        self.camels = self.desert.camels
        self.worms = self.desert.worms
        self.pools = self.desert.pools
        self.window_size = self.desert.size

        self.history = pd.DataFrame(columns=self.history_labels)

    def runViz(self, name, stepsize, max_time=1000, fps=60, save=False):
        pygame.init()
        screen = pygame.display.set_mode(self.window_size)
        clock = pygame.time.Clock()
        font = pygame.font.SysFont('Arial', 20)

        done_clock = 100
        frames = []

        self.running = True
        t = 0
        while self.running and t < max_time:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            if isinstance(self.bg, tuple):
                screen.fill(self.bg)

            if True:
                self.desert.updateAgents(stepsize)

            for worm in self.worms:
                self.drawWorm(screen, worm)
            for camel in self.camels:
                self.drawCamel(screen, camel)


            pygame.display.flip()
            clock.tick(fps)
            t += stepsize

        pygame.quit()

    def drawWorm(self, screen, body: worm.Worm):
        for segment_shapes in body.manifest():
            for shape in segment_shapes:
                shape_type = shape[0]
                color = shape[1]
                position = shape[2]

                if shape_type == 'rect':
                    width, height = shape[3], shape[4]
                    angle = shape[-1]  # Assuming the angle is passed as the last argument in the shape tuple

                    # Create a surface for the rectangle
                    rect_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                    pygame.draw.rect(rect_surface, color, (0, 0, width, height))

                    # Rotate the rectangle surface
                    rotated_surface = pygame.transform.rotate(rect_surface, -angle * 180 / np.pi)

                    # Adjust the position to the center after rotation
                    rotated_rect = rotated_surface.get_rect(center=position)
                    screen.blit(rotated_surface, rotated_rect.topleft)

                elif shape_type == 'circle':
                    radius = shape[3]
                    pygame.draw.circle(screen, color, position, radius)

                elif shape_type == 'polygon':
                    points = shape[3]
                    angle = shape[-1]

                    # Apply rotation to polygon points
                    rotated_points = [
                        (
                            position[0] + (point[0] - position[0]) * np.cos(angle) - (point[1] - position[1]) * np.sin(
                                angle),
                            position[1] + (point[0] - position[0]) * np.sin(angle) + (point[1] - position[1]) * np.cos(
                                angle)
                        )
                        for point in points
                    ]
                    pygame.draw.polygon(screen, color, rotated_points)

    def drawCamel(self, screen, body: camel.Camel):
        for shape in body.manifest():
            shape_type = shape[0]
            color = shape[1]
            position = shape[2]

            if shape_type == 'rect':
                width, height = shape[3], shape[4]
                angle = shape[-1]  # Assuming the angle is passed as the last argument in the shape tuple

                # Create a surface for the rectangle
                rect_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                pygame.draw.rect(rect_surface, color, (0, 0, width, height))

                # Rotate the rectangle surface
                rotated_surface = pygame.transform.rotate(rect_surface, -angle * 180 / np.pi)

                # Adjust the position to the center after rotation
                rotated_rect = rotated_surface.get_rect(center=position)
                screen.blit(rotated_surface, rotated_rect.topleft)

            elif shape_type == 'circle':
                radius = shape[3]
                pygame.draw.circle(screen, color, position, radius)

            elif shape_type == 'polygon':
                points = shape[3]
                angle = shape[-1]

                # Apply rotation to polygon points
                rotated_points = [
                    (
                        position[0] + (point[0] - position[0]) * np.cos(angle) - (point[1] - position[1]) * np.sin(
                            angle),
                        position[1] + (point[0] - position[0]) * np.sin(angle) + (point[1] - position[1]) * np.cos(
                            angle)
                    )
                    for point in points
                ]
                pygame.draw.polygon(screen, color, rotated_points)




if __name__ == '__main__':
    camel = camel.Camel((100, 100), 0.0)
    worm = worm.Worm((400, 400), 0.0)
    desert = desert.Desert((800, 800))
    desert.addWorm(worm)
    desert.addCamel(camel)
    sim = Simulation(desert)
    sim.runViz('test', 1)