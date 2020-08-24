import cv2
import numpy as np


def calibration_process():
    print("No camera calibration detected. Calibration process started...")
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (6, 9)
    CHECKER_SIZE = float(input("Size of Checkerboard Squares in mm: "))  # mm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * CHECKER_SIZE

    capture = cv2.VideoCapture(0)
    images_captured = 0
    print("Hold checkerboard up to camera")
    while True:
        _, img = capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)
        if ret:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            images_captured += 1
            print("Captured %s/20 images" % images_captured)

        cv2.imshow('img', img)
        if images_captured >= 20:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            np.savez("calibration.npz", name1=mtx, name2=dist, name3=rvecs, name4=tvecs)
            print("Calibration Complete!")
            cv2.destroyAllWindows()
            return mtx, dist, rvecs, tvecs

        if cv2.waitKey(10) & 0xFF == ord("q") or images_captured >= 20:
            cv2.destroyAllWindows()
            break
    print("Calibration Failed!")


def get_calibration():
    try:
        camera_data = np.load("calibration.npz")
        camera_matrix = camera_data["name1"]
        camera_dist = camera_data["name2"]
        camera_rvecs = camera_data["name3"]
        camera_tvecs = camera_data["name4"]
        return camera_matrix, camera_dist, camera_rvecs, camera_tvecs
    except:
        return calibration_process()