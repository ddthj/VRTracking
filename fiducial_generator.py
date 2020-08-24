import cv2
import numpy as np


def get_fiducials():
    f_size = 26.5
    fid_points = np.array([
        [0, 0, 0],
        [f_size, 0, 0],
        [f_size, f_size, 0],
        [0, f_size, 0]
    ])
    return cv2.aruco.Dictionary_create(15, 3), cv2.aruco.DetectorParameters_create(), fid_points


def save_fiducials():
    # second parameter is id number
    # last parameter is total image size
    for i in range(15):
        img = cv2.aruco.drawMarker(get_fiducials(), i, 100)
        cv2.imwrite("fiducials/fid_%s.jpg" % i, img)
