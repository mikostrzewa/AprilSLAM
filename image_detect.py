import cv2
import numpy as np
from apriltag import apriltag

# Load the camera calibration data
with np.load('camera_calibration_parameters.npz') as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# Define the actual size of the AprilTag in meters
tag_size = 0.06  # Example: 6 cm

# Initialize the AprilTag detector
detector = apriltag("tagStandard41h12")

# Load the AprilTag image
apriltag_img = cv2.imread('tag0.png')

# Add white padding around the tag
padding = 50
white_padded_image = cv2.copyMakeBorder(apriltag_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

# Add purple padding around the white padded image
purple_padding = 50
final_image = cv2.copyMakeBorder(white_padded_image, purple_padding, purple_padding, purple_padding, purple_padding, cv2.BORDER_CONSTANT, value=[128, 0, 128])

# Convert to grayscale for AprilTag detection
gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
print(gray)
# Detect AprilTags in the image
detections = detector.detect(gray)

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
            cv2.line(final_image, pt1, pt2, (0, 255, 0), 2)

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
        cv2.putText(final_image, text, (pt1[0], pt1[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(final_image, text2, (pt1[0], pt1[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # Draw the coordinate axes on the tag
        cv2.drawFrameAxes(final_image, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

# Display the resulting image
cv2.imshow('AprilTag Detection on Static Image', final_image)

# Wait until any key is pressed to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
