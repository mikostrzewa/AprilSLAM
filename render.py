import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image, ImageOps
import numpy as np
import cv2
from apriltag import apriltag  # Use the same import as in your code

def load_texture(image_name):
    # Load image using PIL
    img = Image.open(image_name)

    # Add white padding around the image
    padding = 50  # Adjust as needed
    img_with_padding = ImageOps.expand(img, border=padding, fill='white')

    # Convert to RGBA format
    img_data = img_with_padding.convert("RGBA").tobytes()
    width, height = img_with_padding.size

    # Generate texture ID and bind it
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Upload texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    return texture_id, width, height

def draw_quad(position, texture_id, quad_size):
    x, y, z = position
    width, height = quad_size
    half_width = width / 200.0  # Scale down for OpenGL units
    half_height = height / 200.0

    glPushMatrix()
    glTranslatef(x, y, z)

    # Bind the texture
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glEnable(GL_TEXTURE_2D)

    # Draw the quad
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex3f(-half_width, -half_height, 0)
    glTexCoord2f(1, 0); glVertex3f(half_width, -half_height, 0)
    glTexCoord2f(1, 1); glVertex3f(half_width, half_height, 0)
    glTexCoord2f(0, 1); glVertex3f(-half_width, half_height, 0)
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glPopMatrix()

def capture_scene(width, height):
    # Read pixels from the OpenGL buffer
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE)
    # Convert to numpy array
    image = np.frombuffer(data, dtype=np.uint8)
    image = image.reshape((height, width, 3))
    # Flip the image vertically
    image = np.flipud(image)
    return image

def main():
    # Load the camera calibration data
    with np.load('camera_calibration_parameters.npz') as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']

    # Define the actual size of the AprilTag in meters
    tag_size = 0.06  # Example: 6 cm

    # Initialize the AprilTag detector using the same method as your code
    detector = apriltag("tagStandard41h12")

    # Initialize Pygame and set up the OpenGL context
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Set up the camera
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, display[0] / display[1], 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Enable depth test
    glEnable(GL_DEPTH_TEST)

    # Set the background color to purple (high contrast)
    glClearColor(128/255.0, 0.0, 128/255.0, 1.0)

    # Load the PNG images as textures with white padding
    textures = []
    for img_name in ['tag0.png', 'tag1.png', 'tag2.png']:
        texture_id, width, height = load_texture(img_name)
        textures.append((texture_id, (width, height)))

    # Define positions for the three images, spread out in 3D space
    positions = [
        (-5.0, -1.0, -15.0),  # Left, further away
        (5.0, 3.0, -10.0),    # Center, closer and higher
        (5.0, -2.0, -20.0)    # Right, furthest away and lower
    ]

    # Print world positions relative to the camera
    print("World Positions relative to the camera:")
    for i, pos in enumerate(positions):
        print(f"Image {i+1}: x={pos[0]}, y={pos[1]}, z={pos[2]}")

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen and depth buffer with the background color
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Draw each image at its position
        for (texture_id, quad_size), position in zip(textures, positions):
            draw_quad(position, texture_id, quad_size)

        # Update the display
        pygame.display.flip()

        # Capture the rendered scene
        image = capture_scene(*display)

        # Convert to grayscale for AprilTag detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags in the image
        detections = detector.detect(gray)
        print(gray)
        print(detections)

        # Process each detected tag
        for detection in detections:
            # Extract corner points of the detected AprilTag
            corners = np.array(detection['lb-rb-rt-lt'], dtype=np.float32)

            # Define the 3D coordinates of the tag's corners in the tag's coordinate frame
            obj_points = np.array([[-tag_size / 2, -tag_size / 2, 0],
                                   [ tag_size / 2, -tag_size / 2, 0],
                                   [ tag_size / 2,  tag_size / 2, 0],
                                   [-tag_size / 2,  tag_size / 2, 0]], dtype=np.float32)

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
                cv2.putText(image, text, (int(corners[0][0]), int(corners[0][1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(image, text2, (int(corners[0][0]), int(corners[0][1]) - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # Draw the coordinate axes on the tag
                cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                # Print the position and orientation
                print(f"Tag ID: {tag_id} - Position (x, y, z): {tvec.flatten()} - Distance: {distance_tvec:.1f} mm")
                print(f"Yaw: {yaw:.1f}, Pitch: {pitch:.1f}, Roll: {roll:.1f}")

        # Display the resulting image
        cv2.imshow('AprilTag Detection', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    # Clean up
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
