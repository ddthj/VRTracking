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


def rotation_matrix(pitch, yaw, roll):
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cr = math.cos(roll)
    sr = math.sin(roll)
    return np.array([[cp * cy, cp * sy, sp],
                     [cy * sp * sr - cr * sy, sy * sp * sr + cr * cy, -cp * sr],
                     [-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr]])


def calibrate(tracked: TrackedObject):
    print("Object '%s' not calibrated. Calibration will begin" % tracked.name)
    # count = int(input("Please enter the number of fiducials on the object: "))
    tags, parems, fid_points = get_fiducials()
    camera_matrix, camera_dist, camera_rvecs, camera_tvecs = get_calibration()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # corners, ids, rejectedimgpoints
        corner_list, ids, _ = cv2.aruco.detectMarkers(gray, tags, parameters=parems)
        found_link = False
        # verify that we actually found fiducials
        if ids is not None:
            ids = [str(x[0]) for x in ids]
            corner_list = [x[0] for x in corner_list]
            # For each fiducial we can see
            for i in range(len(ids)):
                # Get a list of fiducial ids that have already been linked back to the reference
                keys = tracked.fiducials.keys()
                ident = ids[i]
                corners = corner_list[i]
                center = tuple(np.mean(corners, axis=0))
                frame = cv2.putText(frame, ident, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
                if tracked.reference is None:
                    # If our tracked object lacks a reference point, we make this one the reference
                    tracked.reference = ident
                    tracked.fiducials[ident] = Fiducial(ident, None, None)
                    print("selected %s as reference" % ident)
                elif ident not in keys:
                    print("%s not in keys, will attempt to link" % ident)
                    # If this fiducial hasn't been linked yet we get its orientation matrix
                    ident_ret, ident_rot, ident_vec = cv2.solvePnP(fid_points, corners, camera_matrix, camera_dist,
                                                                   cv2.SOLVEPNP_ITERATIVE)
                    ident_vec = ident_vec.reshape(3)
                    ident_matrix = rotation_matrix(*ident_rot)
                    if tracked.reference in ids:
                        print("reference is still visible, making direct link")
                        # If we can see the reference, we get its orientation matrix
                        ref_corners = corner_list[ids.index(tracked.reference)]
                        ref_ret, ref_rot, ref_vec = cv2.solvePnP(fid_points, ref_corners, camera_matrix, camera_dist,
                                                                 cv2.SOLVEPNP_ITERATIVE)
                        ref_vec = ref_vec.reshape(3)
                        ref_matrix = rotation_matrix(*ref_rot)
                        found_link = True

                    else:
                        print("reference isn't visible, searching for another linked fiducial")
                        # We can't see the reference, so instead we must find another linked fiducial
                        for j in range(len(ids)):
                            if j != i and ids[j] in keys:
                                linked = tracked.fiducials[ids[j]]
                                print("selected %s to make indirect link" % linked.ident)
                                linked_corners = corner_list[ids.index(linked.ident)]
                                linked_ret, linked_rot, linked_vec = cv2.solvePnP(fid_points, linked_corners,
                                                                                  camera_matrix,
                                                                                  camera_dist,
                                                                                  cv2.SOLVEPNP_ITERATIVE)
                                linked_vec = linked_vec.reshape(3)
                                linked_matrix = rotation_matrix(*linked_rot)
                                # Once we have the linked fiducial's information we can solve for the reference fiducial
                                ref_matrix = np.dot(linked_matrix, linked.rotation)
                                ref_vec = np.dot(linked.translation, linked.rotation.T) + linked_vec
                                found_link = True
                                break
                    if found_link:
                        # Now that we have the reference pose we can link the new fiducial to it
                        # We find the transformation to get from the current orientation to the reference's orientation
                        ident_to_ref_matrix = np.dot(np.linalg.inv(ident_matrix), ref_matrix)
                        relative = ref_vec - ident_vec
                        local_distance = np.dot(relative, ident_matrix)
                        tracked.fiducials[ident] = Fiducial(ident, ident_to_ref_matrix, local_distance)

                        fid = tracked.fiducials[ident]
                        print("successfully linked %s" % ident)
                        print("ref location:")
                        print(ref_vec)
                        print("estimated ref location:")
                        print(np.dot(fid.translation, fid.rotation.T) + ident_vec)
                        print("ref orientation:")
                        print(ref_matrix)
                        print("estimated ref orientation:")
                        print(np.dot(ident_matrix, fid.rotation))
                    else:
                        print("couldn't find a linked fiducial!")

                #Debug stuff
                elif ident != tracked.reference and tracked.reference in ids:
                    fid = tracked.fiducials[ident]
                    ref_corners = corner_list[ids.index(tracked.reference)]
                    ref_ret, ref_rot, ref_vec = cv2.solvePnP(fid_points, ref_corners, camera_matrix, camera_dist,
                                                             cv2.SOLVEPNP_ITERATIVE)
                    ref_vec = ref_vec.reshape(3)
                    ref_matrix = rotation_matrix(*ref_rot)

                    ident_ret, ident_rot, ident_vec = cv2.solvePnP(fid_points, corners, camera_matrix, camera_dist,
                                                                   cv2.SOLVEPNP_ITERATIVE)
                    ident_vec = ident_vec.reshape(3)
                    ident_matrix = rotation_matrix(*ident_rot)

                    print(ref_vec, np.dot(fid.translation, fid.rotation.T) + ident_vec)
                elif ident != tracked.reference:
                    fid = tracked.fiducials[ident]
                    ident_ret, ident_rot, ident_vec = cv2.solvePnP(fid_points, corners, camera_matrix, camera_dist,
                                                                   cv2.SOLVEPNP_ITERATIVE)
                    ident_vec = ident_vec.reshape(3)
                    print(np.dot(fid.translation, fid.rotation.T) + ident_vec)

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
