import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import cv2
from apriltag import apriltag

def loadTexture(image, padding):
    textureSurface = pygame.image.load(image).convert_alpha()
    width = textureSurface.get_width()
    height = textureSurface.get_height()
    # Create a new surface with white background for padding
    paddedWidth = width + padding * 2
    paddedHeight = height + padding * 2
    paddedSurface = pygame.Surface((paddedWidth, paddedHeight), pygame.SRCALPHA)
    paddedSurface.fill((255, 255, 255, 255))  # White background
    # Blit the original image onto the padded surface
    paddedSurface.blit(textureSurface, (padding, padding))
    textureData = pygame.image.tostring(paddedSurface, "RGBA", True)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, paddedWidth, paddedHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    return texture, paddedWidth, paddedHeight

def main():
    pygame.init()
    display = (1000, 1000)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glEnable(GL_TEXTURE_2D)
    padding = 50  # Amount of padding in pixels

    # Load textures for tag0.png, tag1.png, and tag2.png
    texture0, _, _ = loadTexture('tag0.png', padding)
    texture1, _, _ = loadTexture('tag1.png', padding)
    texture2, _, _ = loadTexture('tag2.png', padding)
    
    # Set up a 3D perspective projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Define positions for each tag in 3D space
    tag_positions = [
        (texture0, (0, 0, -20)),    # tag0 at center
        (texture1, (-2, 0, -10)),   # tag1 to the left
        (texture2, (5, 0, -15)),    # tag2 to the right
    ]
    
    # Set up the camera intrinsic parameters
    fov_y = 45  # Vertical field of view in degrees
    fov_y_rad = np.deg2rad(fov_y)
    height = display[1]
    width = display[0]
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
    tag_size = 0.06  # Actual size of the tag in meters

    # Initialize the AprilTag detector
    detector = apriltag("tagStandard41h12")
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        glClearColor(0.5, 0.0, 0.5, 1.0)  # Clear to purple background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
       
        for texture, position in tag_positions:
            glLoadIdentity()
            glTranslatef(*position)
            glBindTexture(GL_TEXTURE_2D, texture)
            # Render textured quad
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex3f(-1, -1, 0)
            glTexCoord2f(1, 0); glVertex3f(1, -1, 0)
            glTexCoord2f(1, 1); glVertex3f(1, 1, 0)
            glTexCoord2f(0, 1); glVertex3f(-1, 1, 0)
            glEnd()

        # Capture the rendered image
        width, height = display
        data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        image = np.flipud(image)  # Flip the image vertically
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Convert to grayscale for AprilTag detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags in the frame
        detections = detector.detect(gray)

        # Process each detected tag
        for detection in detections:
            # Extract corner points of the detected AprilTag
            corners = np.array(detection['lb-rb-rt-lt'], dtype=np.float32)

            # Define the 3D coordinates of the tag's corners in the tag's coordinate frame
            obj_points = np.array([
                [-tag_size / 2,  tag_size / 2, 0],  # Top-left corner
                [ tag_size / 2,  tag_size / 2, 0],  # Top-right corner
                [ tag_size / 2, -tag_size / 2, 0],  # Bottom-right corner
                [-tag_size / 2, -tag_size / 2, 0],  # Bottom-left corner
            ], dtype=np.float32)

            # Estimate the pose of the tag
            retval, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)

            if retval:
                # Draw the detected tag corners on the image
                for i in range(4):
                    pt1 = tuple(map(int, corners[i]))
                    pt2 = tuple(map(int, corners[(i + 1) % 4]))
                    cv2.line(image, pt1, pt2, (0, 255, 0), 2)

                # Convert rotation vector to rotation matrix
                rot_matrix, _ = cv2.Rodrigues(rvec)

                # Calculate yaw, pitch, and roll from the rotation matrix
                sy = np.sqrt(rot_matrix[0, 0] ** 2 + rot_matrix[1, 0] ** 2)
                singular = sy < 1e-6
                if not singular:
                    yaw = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
                    pitch = np.arctan2(-rot_matrix[2, 0], sy)
                    roll = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
                else:
                    yaw = np.arctan2(-rot_matrix[1, 2], rot_matrix[1, 1])
                    pitch = np.arctan2(-rot_matrix[2, 0], sy)
                    roll = 0

                # Convert angles to degrees
                yaw, pitch, roll = np.degrees([yaw, pitch, roll])

                # Calculate the distance using the translation vector and convert to millimeters
                distance_tvec = np.linalg.norm(tvec) * 1000  # Convert to mm

                # Display the tag's information on the image
                tag_id = detection['id']
                text = f'ID: {tag_id}, Dist: {distance_tvec:.1f} mm'
                text2 = f'Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}'
                cv2.putText(image, text, (pt1[0], pt1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(image, text2, (pt1[0], pt1[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Draw the coordinate axes on the tag
                cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                # Print the position and distances
                print(f"Tag ID: {tag_id} - Position (x, y, z): {tvec.flatten()} - Distance (tvec): {distance_tvec:.1f} mm")
                print(f"Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}")

        # Display the resulting image with detections
        cv2.imshow('AprilTag Detection', image)

        # Swap the OpenGL buffers
        pygame.display.flip()
        pygame.time.wait(10)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()
        
if __name__ == '__main__':
    main()
