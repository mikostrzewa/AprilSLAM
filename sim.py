import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2
from apriltag import apriltag
from slam import SLAM

# Constants
DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 1000
FOV_Y = 45  # Vertical field of view in degrees
NEAR_CLIP = 0.1
FAR_CLIP = 100.0


def loadTexture(image):
    textureSurface = pygame.image.load(image).convert_alpha()
    width = textureSurface.get_width()
    height = textureSurface.get_height()
    textureData = pygame.image.tostring(textureSurface, "RGBA", True)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    return texture, width, height

def main():
    pygame.init()
    display = (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glEnable(GL_TEXTURE_2D)
    
    # Load textures for tag0.png, tag1.png, and tag2.png
    texture0, _, _ = loadTexture('tags/tag0.png')
    texture1, _, _ = loadTexture('tags/tag1.png')
    texture2, _, _ = loadTexture('tags/tag2.png')
    
    # Set up a 3D perspective projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(FOV_Y, (DISPLAY_WIDTH / DISPLAY_HEIGHT), NEAR_CLIP, FAR_CLIP)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Define positions and rotations for each tag in 3D space
    tag_positions = [
        (texture0, (0, 0, -70), (0, 0, 0)),    # tag0 at center with no rotation
        (texture1, (-27, 3, -90), (0, 0, 0)),   # tag1 to the left with 45 degrees rotation around x-axis
        (texture2, (23, -2, -80), (0, 0, 0)),    # tag2 to the right with 45 degrees rotation around y-axis
    ]
    
    # Initialize SLAM
    # Set up the camera intrinsic parameters
    fov_y_rad = np.deg2rad(FOV_Y)
    height = DISPLAY_HEIGHT
    width = DISPLAY_WIDTH
    fy = (height / 2) / np.tan(fov_y_rad / 2)
    fx = fy * (width / height)
    cx = width / 2
    cy = height / 2
    camera_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)
    # Calculate the tag size in world render units
    tag_size = 10  # Adjust this value to match the actual size in your render units

    camera_params = {"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs}
    slam = SLAM(camera_params,tag_size=tag_size)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        glClearColor(0.5, 0.0, 0.5, 1.0)  # Clear to purple background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
       
        for texture, position, rotation in tag_positions:
            glLoadIdentity()
            glTranslatef(*position)
            glRotatef(rotation[0], 1, 0, 0)  # Rotate around x-axis
            glRotatef(rotation[1], 0, 1, 0)  # Rotate around y-axis
            glRotatef(rotation[2], 0, 0, 1)  # Rotate around z-axis
            glBindTexture(GL_TEXTURE_2D, texture)
            # Render textured quad with size 18x18
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex3i(-9, -9, 0)
            glTexCoord2f(1, 0); glVertex3i(9, -9, 0)
            glTexCoord2f(1, 1); glVertex3i(9, 9, 0)
            glTexCoord2f(0, 1); glVertex3i(-9, 9, 0)
            glEnd()
            
            # Get the actual texture size from OpenGL
            actual_width = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
            actual_height = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)

        # Capture the rendered image
        width, height = display
        data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        image = np.flipud(image)  # Flip the image vertically
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect AprilTags in the frame
        detections = slam.detect(image)
        
        # Process each detected tag
        for detection in detections:
            retval, rvec, tvec = slam.get_pose(detection)
            if retval:
                corners = np.array(detection['lb-rb-rt-lt'], dtype=np.float32)
                image = slam.draw(rvec, tvec, corners, image, detection['id'])
        
        # Display the resulting image with detections
        cv2.imshow('AprilTag Detection', image)

        pose = slam.my_pose()
        if pose is not None:
            translation_vector = pose[:3, 3]
            rotation_matrix = pose[:3, :3]
            print("Translation Vector:", translation_vector)
            print("Rotation Matrix:\n", rotation_matrix)
        
        # Swap the OpenGL buffers
        pygame.display.flip()
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
