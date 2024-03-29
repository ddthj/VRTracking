import math

import cv2
import numpy as np
from fiducial_generator import get_fiducials
from calibrate_camera import get_calibration

s = 13.25
FIDUCIAL_CORNERS = np.float32([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]]).reshape(-1, 3)
FIDUCIAL_CORNERS_DEPTH = np.float32([[-s, s, -16], [s, s, -16], [s, -s, -16], [-s, -s, -16]]).reshape(-1, 3)
FIDUCIAL_CENTER = np.array([s, s, 0])


def npv(x, y, z):
    return np.float32([x, y, z]).reshape(3)


HEADSET_FIDUCIALS = {
    0: npv(0, 0, -150.0),
    1: npv(-31, -35.6, -150.0),
    2: npv(32.6, -35.6, -150.0),
    4: npv(0, 21.8, -113.5),
    5: npv(-45, 21.8, -127.7),
    6: npv(45, 21.8, -127.7),
    7: npv(-75, -3.5, -22.3),
    8: npv(-75, -40.1, -22.3),
    9: npv(-93.3, -20, -122.7),
    10: npv(75, -40.1, -22.3),
    11: npv(75, -3.5, -22.3),
    13: npv(-93.3, -13.5, -124)
}


class TrackedObject:
    def __init__(self, name):
        self.name = name
        self.fiducials = {}
        self.reference = None


class Fiducial:
    def __init__(self, ident, rotation, translation):
        self.ident = ident
        self.rotation = rotation
        self.translation = translation
        self.samples = [translation]

    def get_loc(self):
        temp_avg = sum(self.samples) / len(self.samples)
        final = []
        for item in self.samples:
            if np.linalg.norm(item - temp_avg) < 20:
                final.append(item)
        if len(final) > 0:
            return sum(final) / len(final)
        else:
            return temp_avg


def calibrate(tracked: TrackedObject):
    print("Object '%s' not calibrated. Calibration will begin" % tracked.name)
    count = 99  # int(input("Please enter the number of fiducials on the object: "))
    tags, parems, fid_points = get_fiducials()
    camera_matrix, camera_dist = get_calibration()
    cap = cv2.VideoCapture(0)

    while len(tracked.fiducials) < count:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # corners, ids, debug
        corner_list, ids, _ = cv2.aruco.detectMarkers(gray, tags, parameters=parems)
        link = None
        link_is_reference = False
        if ids is not None:
            ids = list(ids.ravel())
            object_points = np.float32([HEADSET_FIDUCIALS[x] for x in ids if HEADSET_FIDUCIALS.get(x, None) is not None])
            centers = []
            for i in range(len(ids)):
                corners = corner_list[i][0]
                centers.append(np.mean(corners, axis=0))

            if len(centers) >= 4:
                try:
                    hmd_ret, hmd_rot, hmd_vec = cv2.solvePnP(object_points, np.float32(centers), camera_matrix, camera_dist)
                    p, _ = cv2.projectPoints(FIDUCIAL_CORNERS, hmd_rot, hmd_vec, camera_matrix,
                                          camera_dist)
                    frame = cv2.polylines(frame, [np.int32(p).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
                except:
                    pass

            """
            for i in range(len(ids)):
                keys = tracked.fiducials.keys()
                ident = ids[i]
                corners = corner_list[i][0]

                if tracked.reference is None:
                    # If our tracked object lacks a reference point, we make this one the reference
                    tracked.reference = ident
                    tracked.fiducials[ident] = Fiducial(ident, None, None)
                    link = tracked.fiducials[ident]
                    link_is_reference = True
                    print("selected %s as reference" % ident)
                elif ident not in keys:
                    print("%s not in keys, will attempt to link" % ident)
                    # To add a new fiducial we first need to identify an existing one, preferably the reference
                    if tracked.reference in ids:
                        print("reference is still visible, making direct link")
                        link = tracked.fiducials[tracked.reference]
                        link_is_reference = True
                    elif link is None:
                        print("reference isn't visible, searching for another linked fiducial")
                        for j in range(len(ids)):
                            if j != i and ids[j] in keys:
                                link = tracked.fiducials[ids[j]]
                                print("selected %s to make indirect link" % ids[j])
                                break

                    if link is not None:
                        # Now that we have a link, we find the pose of the ident and link
                        ident_ret, ident_rot, ident_vec = cv2.solvePnP(fid_points, corners, camera_matrix, camera_dist,
                                                                       cv2.SOLVEPNP_ITERATIVE)
                        ident_matrix = cv2.Rodrigues(ident_rot, np.identity(3), jacobian=0)[0]
                        ident_loc = ident_vec.reshape(3) + FIDUCIAL_CENTER.dot(ident_matrix.T)

                        link_corners = corner_list[ids.index(link.ident)][0]
                        link_ret, link_rot, link_vec = cv2.solvePnP(fid_points, link_corners, camera_matrix,
                                                                    camera_dist,
                                                                    cv2.SOLVEPNP_ITERATIVE)
                        link_matrix = cv2.Rodrigues(link_rot, np.identity(3), jacobian=0)[0]
                        link_loc = link_vec.reshape(3) + FIDUCIAL_CENTER.dot(link_matrix.T)

                        if link_is_reference:
                            ref_matrix = link_matrix
                            ref_loc = link_loc
                        else:
                            ref_matrix = link.rotation.dot(ident_matrix)
                            ref_loc = ident_loc + ident_matrix.dot(link.translation)

                        ident_to_ref_matrix = np.linalg.inv(ident_matrix).dot(ref_matrix)
                        relative = ident_loc - ref_loc
                        local_distance = relative.dot(ref_matrix)
                        tracked.fiducials[ident] = Fiducial(ident, ident_to_ref_matrix, local_distance)
                    else:
                        print("couldn't find a linked fiducial!")

                # Debug stuff
                elif ident != tracked.reference:
                    link = tracked.fiducials[ident]
                    point = np.array([0, 0, 0], dtype=np.float32).reshape(3)
                    ident_ret, ident_rot, ident_vec = cv2.solvePnP(fid_points, corners, camera_matrix, camera_dist,
                                                                   cv2.SOLVEPNP_ITERATIVE)
                    ident_matrix = cv2.Rodrigues(ident_rot, np.identity(3), jacobian=0)[0]
                    ident_loc = ident_vec.reshape(3) + FIDUCIAL_CENTER.dot(ident_matrix.T)

                    projected, _ = cv2.projectPoints(point, ident_rot, ident_vec, camera_matrix, camera_dist)
                    frame = cv2.putText(frame, str(ident), tuple(projected.ravel()), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), thickness=2)
                    projected2, _ = cv2.projectPoints(FIDUCIAL_CORNERS, ident_rot, ident_loc, camera_matrix,
                                                      camera_dist)
                    test, _ = cv2.projectPoints(FIDUCIAL_CORNERS_DEPTH, ident_rot, ident_loc, camera_matrix,
                                                camera_dist)
                    frame = cv2.polylines(frame, [np.int32(test).reshape(-1, 1, 2)], True, (0, 0, 255), 2)
                    cv2.circle(frame, tuple(projected.ravel()), 5, (0, 0, 255), -1)
                    frame = cv2.polylines(frame, [np.int32(projected2).reshape(-1, 1, 2)], True, (0, 0, 255), 2)

                    est_ref_loc = ident_loc + ident_matrix.dot(link.translation)
                    est_ref_rot = ident_matrix.dot(link.rotation)  # link.rotation.T.dot(ident_matrix)
                    projected3, _ = cv2.projectPoints(FIDUCIAL_CORNERS, est_ref_rot, est_ref_loc, camera_matrix,
                                                      camera_dist)
                    frame = cv2.polylines(frame, [np.int32(projected3).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            """

        cv2.imshow("tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Calibration of '%s' complete!" % tracked.name)
