# Computes the 3D positions of the markers on the pen, given a set of images of the pen.
# The images should be taken with the pen in a variety of poses, and each image should contain at least two markers.
# The calibrated positions are stored OUTPUT_PATH

import json
from pathlib import Path
import numpy as np
import cv2
import glob
import scipy.optimize
from cv2 import aruco
from app.marker_tracker import aruco_params, read_camera_parameters
from app.dimensions import idealMarkerPositions

MARKER_COUNT = 8  # The number of markers on the pen
FIRST_MARKER_ID = 92  # ID of the first marker on the pen, used to convert IDs to 0-n.
IMAGE_PATH = "./marker_calibration_pics/f30/*.jpg"
OUTPUT_PATH = "./params/calibrated_marker_positions.json"


Observation = dict[int, np.ndarray]  # Map from marker id to 4x3 array of corners


def residual(
    x: np.ndarray,
    camera_matrix,
    dist_coeffs,
    observations: list[Observation],
    marker0_pose: np.ndarray,
):
    marker_poses = np.concatenate(
        (marker0_pose, x[0 : (MARKER_COUNT - 1) * 4 * 3])
    ).reshape((MARKER_COUNT, 4, 3))
    camera_poses = x[(MARKER_COUNT - 1) * 4 * 3 :].reshape((-1, 6))
    res_all = []
    for img_id in range(len(observations)):
        img = observations[img_id]
        rvec = camera_poses[img_id, 0:3]
        tvec = camera_poses[img_id, 3:6]

        for marker_id, marker_corners_observed in img.items():
            projected: np.ndarray
            projected, jac = cv2.projectPoints(
                objectPoints=marker_poses[marker_id - FIRST_MARKER_ID, :, :],  # 4x3
                rvec=rvec,
                tvec=tvec,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
            )

            res = projected.flatten() - marker_corners_observed.flatten()
            res_all.append(res)

    return np.concatenate(res_all)


def get_observed_points(pathname, arucoParams):
    """Reads in the set of images, and detects any markers in them."""
    observed_points: list[Observation] = []
    for iname in glob.glob(pathname):
        img = cv2.imread(iname)
        corners_all, ids, _ = aruco.detectMarkers(
            image=img,
            dictionary=aruco.getPredefinedDictionary(aruco.DICT_4X4_100),
            parameters=arucoParams,
        )
        if ids is None:
            print("No markers found in image", iname)
            continue
        if ids.shape[0] < 2:
            print("Not enough markers found in image", iname)
            continue
        corner_dict: Observation = {}
        for i in range(ids.shape[0]):
            if ids[i, 0] in idealMarkerPositions:
                corner_dict[ids[i, 0]] = corners_all[i][0, :, :]
        observed_points.append(corner_dict)

    if not observed_points:
        raise RuntimeError("No valid images found in path", pathname)
    return observed_points


def get_initial_estimate(observations: list[Observation], camera_matrix, dist_coeffs):
    """Computes an initial state vector based on:
    - The ideal marker positions
    - Camera poses for each image, estimated using PnP
    """
    marker_poses = np.zeros((MARKER_COUNT, 4, 3), dtype=np.float32)
    for mid, corners in idealMarkerPositions.items():
        marker_poses[mid - FIRST_MARKER_ID, :, :] = corners

    camera_poses = np.zeros((len(observations), 6), dtype=np.float32)
    for img_id, img in enumerate(observations):
        validMarkers = []
        for marker_id, marker_corners_observed in img.items():
            if marker_id in idealMarkerPositions:
                validMarkers.append(
                    (idealMarkerPositions[marker_id], marker_corners_observed)
                )

        screenCorners = np.concatenate([cornersIS for _, cornersIS in validMarkers])
        penCorners = np.concatenate([cornersPS for cornersPS, _ in validMarkers])
        success, rvec, tvec = cv2.solvePnP(
            penCorners,
            screenCorners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        camera_poses[img_id, :] = np.hstack((rvec.flatten(), tvec.flatten()))

    result = np.concatenate((marker_poses.flatten(), camera_poses.flatten()))
    return result[0:12], result[12:]


def calibrate_markers(camera_matrix, dist_coeffs, observations: list[Observation]):
    # We fix the pose of the first marker, so that the problem is properly constrained.
    marker0_pose, x0 = get_initial_estimate(observations, camera_matrix, dist_coeffs)

    def fun(x):
        return residual(x, camera_matrix, dist_coeffs, observations, marker0_pose)

    opt_result = scipy.optimize.least_squares(
        fun=fun,
        x0=x0,
        max_nfev=1000,
        verbose=2,
    )
    return np.concatenate((marker0_pose, opt_result.x[0 : 7 * 4 * 3])).reshape(
        (MARKER_COUNT, 4, 3)
    )


def main():
    observed_points = get_observed_points(IMAGE_PATH, aruco_params)
    camera_matrix, dist_coeffs = read_camera_parameters("./params/camera_params_c922_f30.yml")
    result = calibrate_markers(camera_matrix, dist_coeffs, observed_points)

    calibratedMarkerPositions = {
        i + FIRST_MARKER_ID: result[i, :, :].tolist() for i in range(MARKER_COUNT)
    }
    file = Path(OUTPUT_PATH)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w") as f:
        json.dump(calibratedMarkerPositions, f, indent=2)


if __name__ == "__main__":
    main()
