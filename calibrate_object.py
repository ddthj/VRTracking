import math

import cv2
import numpy as np
from fiducial_generator import get_fiducials
from calibrate_camera import get_calibration

s = 13.25
FIDUCIAL_CORNERS = np.float32([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]]).reshape(-1, 3)
FIDUCIAL_CENTER = np.array([s, s, 0])


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
                        relative = ref_loc - ident_loc
                        print("relative distance: ", relative)
                        local_distance = relative.dot(ident_matrix)
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
                    cv2.circle(frame, tuple(projected.ravel()), 5, (0, 0, 255), -1)
                    frame = cv2.polylines(frame, [np.int32(projected2).reshape(-1, 1, 2)], True, (0, 0, 255), 2)

                    est_ref_loc = ident_loc + ident_matrix.dot(link.translation)
                    est_ref_rot = ident_matrix.dot(link.rotation)  # link.rotation.T.dot(ident_matrix)
                    projected3, _ = cv2.projectPoints(FIDUCIAL_CORNERS, est_ref_rot, est_ref_loc, camera_matrix,
                                                      camera_dist)
                    frame = cv2.polylines(frame, [np.int32(projected3).reshape(-1, 1, 2)], True, (0, 255, 0), 2)

        cv2.imshow("tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Calibration of '%s' complete!" % tracked.name)


"""
                elif ident != tracked.reference:
                    point = np.array([0, 0, 0], dtype=np.float32).reshape(-1, 3)
                    # projected, _ = cv2.projectPoints(point, ident_rot, ident_vec, camera_matrix, camera_dist)
                    # cv2.circle(frame, tuple(projected[0].ravel()), 5, (0, 255, 0), -1)
                    fid = tracked.fiducials[ident]
                    ident_center = ident_vec.reshape(3) + np.dot(np.linalg.inv(ident_matrix), FIDUCIAL_CENTER)
                    mini, __ = cv2.projectPoints(point, ident_rot, ident_center, camera_matrix, camera_dist)
                    frame = cv2.circle(frame, tuple(np.int32(mini).reshape(2).tolist()), 5, (0, 34, 255), -1)

                    ref_vec_est = np.dot(fid.rotation.T, fid.translation) + ident_vec
                    ref_rot_est = np.dot(fid.rotation, ident_matrix)

                    # projected, _ = cv2.projectPoints(FIDUCIAL_CORNERS, ident_rot, ident_vec, camera_matrix, camera_dist)
                    # projected, _ = cv2.projectPoints(FIDUCIAL_CORNERS, ref_rot_est, ref_vec_est, camera_matrix, camera_dist)
                    # frame = cv2.polylines(frame, [np.int32(projected).reshape(-1, 1, 2)], True, (0, 34, 255), 2)
"""
