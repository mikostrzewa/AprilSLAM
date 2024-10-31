import sys
import json
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2
from apriltag import apriltag
from slam import SLAM

class Simulation:
    def __init__(self, settings_file):
        self.load_settings(settings_file)
        self.init_pygame()
        self.load_textures()
        self.init_opengl()
        self.init_slam()

    def load_settings(self, settings_file):
        with open(settings_file, 'r') as f:
            self.settings = json.load(f)
        self.display_width = self.settings["display_width"]
        self.display_height = self.settings["display_height"]
        self.fov_y = self.settings["fov_y"]
        self.near_clip = self.settings["near_clip"]
        self.far_clip = self.settings["far_clip"]
        self.tag_size = self.settings["tag_size"]
        self.tags = self.settings["tags"]

    def init_pygame(self):
        pygame.init()
        self.display = (self.display_width, self.display_height)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        glEnable(GL_TEXTURE_2D)

    def load_textures(self):
        self.textures = []
        for tag in self.tags:
            texture, _, _ = self.load_texture(tag["image"])
            self.textures.append((texture, tag["position"], tag["rotation"]))

    def load_texture(self, image):
        textureSurface = pygame.image.load(image).convert_alpha()
        width = textureSurface.get_width()
        height = textureSurface.get_height()
        textureData = pygame.image.tostring(textureSurface, "RGBA", True)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        return texture, width, height

    def init_opengl(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov_y, (self.display_width / self.display_height), self.near_clip, self.far_clip)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def init_slam(self):
        # Calculate camera parameters
        fx = fy = 0.5 * self.display_height / np.tan(0.5 * np.radians(self.fov_y))
        cx = 0.5 * self.display_width
        cy = 0.5 * self.display_height
        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]])
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        camera_params = {'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs}
        self.slam = SLAM(camera_params,tag_size=self.tag_size)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            glClearColor(0.5, 0.0, 0.5, 1.0)  # Clear to purple background
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            for texture, position, rotation in self.textures:
                glLoadIdentity()
                glTranslatef(*position)
                glRotatef(rotation[0], 1, 0, 0)  # Rotate around x-axis
                glRotatef(rotation[1], 0, 1, 0)  # Rotate around y-axis
                glRotatef(rotation[2], 0, 0, 1)  # Rotate around z-axis
                glBindTexture(GL_TEXTURE_2D, texture)
                # Render textured quad with size based on tag_size
                size = self.tag_size / 2
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0); glVertex3f(-size, -size, 0)
                glTexCoord2f(1, 0); glVertex3f(size, -size, 0)
                glTexCoord2f(1, 1); glVertex3f(size, size, 0)
                glTexCoord2f(0, 1); glVertex3f(-size, size, 0)
                glEnd()
            
            # Capture the rendered image
            width, height = self.display
            data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            image = np.flipud(image)  # Flip the image vertically
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Detect AprilTags in the frame
            detections = self.slam.detect(image)
            
            # Process each detected tag
            for detection in detections:
                retval, rvec, tvec = self.slam.get_pose(detection)
                if retval:
                    corners = np.array(detection['lb-rb-rt-lt'], dtype=np.float32)
                    image = self.slam.draw(rvec, tvec, corners, image, detection['id'])
            
            # Display the resulting image with detections
            cv2.imshow('AprilTag Detection', image)

            pose = self.slam.my_pose()
            if pose is not None:
                translation_vector = pose[:3, 3]
                rotation_matrix = pose[:3, :3]
                print("Translation Vector:", translation_vector)
                print("Rotation Matrix:\n", rotation_matrix)
            
            pygame.display.flip()
            pygame.time.wait(10)

if __name__ == '__main__':
    sim = Simulation('sim_settings.json')
    sim.run()
