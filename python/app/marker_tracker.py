import json
import math
from pathlib import Path
import os
import pickle
import cv2
from cv2 import aruco
import numpy as np
from typing import NamedTuple, Tuple, Callable, Optional
import time
import sys
import multiprocessing as mp

from app.dimensions import IMU_OFFSET, STYLUS_LENGTH, idealMarkerPositions

RECORD_DATA = True
FPS = 30
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
TEXT_COL = (0, 0, 255)


class CameraReading(NamedTuple):
    position: np.ndarray
    orientation_mat: np.ndarray

    def to_json(self):
        return {
            "position": self.position.tolist(),
            "orientation_mat": self.orientation_mat.tolist(),
        }

    def from_json(dict):
        return CameraReading(
            np.array(dict["position"]), np.array(dict["orientation_mat"])
        )


def read_camera_parameters(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise Exception("Couldn't open file")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return (camera_matrix, dist_coeffs)


def get_webcam():
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    webcam.set(cv2.CAP_PROP_FPS, 60)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not webcam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")):
        raise Exception("Couldn't set FourCC")
    webcam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    webcam.set(cv2.CAP_PROP_FOCUS, 30)
    webcam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    webcam.set(cv2.CAP_PROP_EXPOSURE, -9)
    webcam.set(cv2.CAP_PROP_BRIGHTNESS, 127)
    webcam.set(cv2.CAP_PROP_CONTRAST, 140)
    webcam.set(cv2.CAP_PROP_GAIN, 200)
    webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return webcam


def inverse_RT(rvec, tvec) -> Tuple[np.ndarray, np.ndarray]:
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.transpose(R)
    return (cv2.Rodrigues(Rt)[0], -Rt @ tvec)


def relative_transform(rvec1, tvec1, rvec2, tvec2) -> Tuple[np.ndarray, np.ndarray]:
    rvec2inv, tvec2inv = inverse_RT(rvec2, tvec2)
    rvec, tvec, *_ = cv2.composeRT(rvec1, tvec1, rvec2inv, tvec2inv)
    return (rvec, tvec)


def get_aruco_params():
    p = aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    p.cornerRefinementWinSize = 2
    # Reduce the number of threshold steps, which significantly improves performance
    p.adaptiveThreshWinSizeMin = 15
    p.adaptiveThreshWinSizeMax = 15
    p.useAruco3Detection = False
    p.minMarkerPerimeterRate = 0.02
    p.maxMarkerPerimeterRate = 2
    p.minSideLengthCanonicalImg = 16
    p.adaptiveThreshConstant = 7
    return p


aruco_dic = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
aruco_params = get_aruco_params()
detector = aruco.ArucoDetector(aruco_dic, aruco_params)

reprojection_error_threshold = 3  # px


def array_to_str(arr):
    return ",".join(map(lambda x: f"{x:+2.2f}", list(arr.flat)))


# charuco_dic = aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
# charuco_board = aruco.CharucoBoard((10, 7), 0.028, 0.022, charuco_dic)
charuco_dic = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
charuco_board = aruco.CharucoBoard((12, 8), 0.024, 0.018, charuco_dic)
charuco_board.setLegacyPattern(True)
charuco_params = aruco.DetectorParameters()
charuco_detector = aruco.ArucoDetector(charuco_dic, charuco_params)


def estimate_camera_pose_charuco(frame, camera_matrix, dist_coeffs):
    corners, ids, rejected = charuco_detector.detectMarkers(frame)
    if len(corners) == 0:
        raise Exception("No markers detected")
    display_frame = aruco.drawDetectedMarkers(image=frame, corners=corners)
    num_corners, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=corners, markerIds=ids, image=frame, board=charuco_board
    )
    if num_corners < 5:
        raise Exception("Not enough corners detected")
    display_frame = aruco.drawDetectedCornersCharuco(
        image=display_frame, charucoCorners=charuco_corners, charucoIds=charuco_ids
    )
    success, rvec, tvec = aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        charuco_board,
        camera_matrix,
        dist_coeffs,
        None,
        None,
        False,
    )
    if not success:
        raise Exception("Failed to estimate camera pose")
    # The rvec from charuco is z-down for some reason.
    # This is a hack to convert back to z-up.
    rvec, *_ = cv2.composeRT(np.array([0, 0, -np.pi / 2]), tvec * 0, rvec, tvec)
    rvec, *_ = cv2.composeRT(np.array([0, np.pi, 0]), tvec * 0, rvec, tvec)
    display_frame = cv2.drawFrameAxes(
        display_frame, camera_matrix, dist_coeffs, rvec, tvec, 0.2
    )
    # cv2.imshow("Charuco", display_frame)
    return (rvec, tvec)


def vector_rms(arr: np.ndarray, axis: int):
    """Computes the RMS magnitude of an array of vectors."""
    return math.sqrt(np.mean(np.sum(np.square(arr), axis=axis)))


def solve_pnp(
    initialized,
    prev_rvec,
    prev_tvec,
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Attempt to refine the previous pose. If this fails, fall back to SQPnP."""
    if initialized:
        rvec, tvec = cv2.solvePnPRefineVVS(
            object_points,
            image_points,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            # OpenCV mutates these arguments, which we don't want.
            rvec=prev_rvec.copy(),
            tvec=prev_tvec.copy(),
        )
        projected_image_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, dist_coeffs, None
        )
        projected_image_points = projected_image_points[:, 0, :]
        reprojection_error = vector_rms(projected_image_points - image_points, axis=1)

        if reprojection_error < reprojection_error_threshold:
            # print(f"Reprojection error: {reprojectionError}")
            return (True, rvec, tvec)
        else:
            print(f"Reprojection error too high: {reprojection_error}")

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    return (success, rvec, tvec)


MarkerDict = dict[int, tuple[np.ndarray, np.ndarray]]


def detect_markers_bounded(frame: np.ndarray, x0: int, x1: int, y0: int, y1: int):
    x0, y0 = max(x0, 0), max(y0, 0)
    frame_view = frame[y0:y1, x0:x1]
    ids = None
    allCornersIS = []
    rejected = []
    try:
        allCornersIS, ids, rejected = detector.detectMarkers(frame_view)
    except cv2.error as e:
        # OpenCV threw an error here once for some reason, but we'd rather ignore it.
        # D:\a\opencv-python\opencv-python\opencv\modules\objdetect\src\aruco\aruco_detector.cpp:698: error: (-215:Assertion failed) nContours.size() >= 2 in function 'cv::aruco::_interpolate2Dline'
        print(e)
        pass
    if ids is not None:
        for i in range(ids.shape[0]):
            allCornersIS[i][0, :, 0] += x0
            allCornersIS[i][0, :, 1] += y0
    return allCornersIS, ids, rejected


def bounds(x):
    return np.min(x), np.max(x)


def clamp(x, xmin, xmax):
    return max(min(x, xmax), xmin)


class MarkerTracker:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        marker_positions: dict[int, np.ndarray],
    ):
        self.cameraMatrix = camera_matrix
        self.distCoeffs = dist_coeffs
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None
        self.initialized = False
        self.markerPositions = marker_positions
        self.allObjectPoints = np.concatenate(list(marker_positions.values()))
        self.lastValidMarkers: MarkerDict = {}
        self.lastVelocity = np.zeros(2)

    def get_search_area(self, rvec: np.ndarray, tvec: np.ndarray, velocity: np.array):
        """Returns a bounding box to search in the next frame, based on the current marker positions and velocity."""

        # Re-project all object points, to avoid cases where some markers were missed in the previous frame.
        projected_image_points, _ = cv2.projectPoints(
            self.allObjectPoints, rvec, tvec, self.cameraMatrix, self.distCoeffs, None
        )
        projected_image_points = projected_image_points[:, 0, :]

        x0, x1 = bounds(projected_image_points[:, 0] + velocity[0] / FPS)
        y0, y1 = bounds(projected_image_points[:, 1] + velocity[1] / FPS)
        w = x1 - x0
        h = y1 - y0

        # Amount to expand each axis by, in pixels. This is just a rough heuristic, and the constants are arbitrary.
        expand = max(0.5 * (w + h), 200) + 1.0 * np.abs(velocity) / FPS

        # Values are sometimes extremely large if tvec is wrong, clamp is a workaround to stop cv2.rectangle from breaking.
        return (
            int(clamp(x0 - expand[0], 0, FRAME_WIDTH)),
            int(clamp(x1 + expand[0], 0, FRAME_WIDTH)),
            int(clamp(y0 - expand[1], 0, FRAME_HEIGHT)),
            int(clamp(y1 + expand[1], 0, FRAME_HEIGHT)),
        )

    def process_frame(self, frame: np.ndarray):
        ids: np.ndarray
        if self.initialized:
            x0, x1, y0, y1 = self.get_search_area(
                self.rvec, self.tvec, self.lastVelocity
            )
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 100, 0, 0.3), 2)
            allCornersIS, ids, rejected = detect_markers_bounded(frame, x0, x1, y0, y1)
        else:
            allCornersIS, ids, rejected = detector.detectMarkers(frame)
        aruco.drawDetectedMarkers(frame, allCornersIS, ids)
        valid_markers: MarkerDict = {}
        if ids is not None:
            for i in range(ids.shape[0]):
                # cornersIS is 4x2
                id, cornersIS = (ids[i, 0], allCornersIS[i][0, :, :])
                if id in self.markerPositions:
                    cornersPS = self.markerPositions[id]
                    valid_markers[id] = (cornersPS, cornersIS)

        if len(valid_markers) < 1:
            self.initialized = False
            self.lastValidMarkers = {}
            self.next_search_area = None
            return None

        point_deltas = []
        for id, (cornersPS, cornersIS) in valid_markers.items():
            if id in self.lastValidMarkers:
                velocity = cornersIS - self.lastValidMarkers[id][1]
                point_deltas.append(np.mean(velocity, axis=0))

        if point_deltas:
            meanVelocity = np.mean(point_deltas, axis=0) * 30  # px/second
        else:
            meanVelocity = np.zeros(2)

        mean_position_IS = np.mean(
            [cornersIS for _, cornersIS in valid_markers.values()],
            axis=(0, 1),
        )

        screen_corners = []
        pen_corners = []
        delay_per_image_row = 1 / 30 / 1080  # seconds/row

        for id, (cornersPS, cornersIS) in valid_markers.items():
            pen_corners.append(cornersPS)
            if point_deltas:
                # Compensate for rolling shutter
                timeDelay = (
                    cornersIS[:, 1] - mean_position_IS[1]
                ) * delay_per_image_row  # seconds, relative to centroid
                cornersISCompensated = (
                    cornersIS - meanVelocity * timeDelay[:, np.newaxis]
                )
                screen_corners.append(cornersISCompensated)
            else:
                screen_corners.append(cornersIS)

        self.initialized, self.rvec, self.tvec = solve_pnp(
            self.initialized,
            self.rvec,
            self.tvec,
            object_points=np.concatenate(pen_corners),
            image_points=np.concatenate(screen_corners),
            camera_matrix=self.cameraMatrix,
            dist_coeffs=self.distCoeffs,
        )

        self.lastValidMarkers = valid_markers
        self.lastVelocity = meanVelocity
        return (self.rvec, self.tvec)


focus_interval = 30  # frames
# Map from distance to optimal focus value, measured manually.
# These don't need to be very precise.
focus_targets = np.array(
    [
        [0.1, 75],
        [0.15, 50],
        [0.2, 40],
        [0.3, 30],
        [0.5, 25],
    ]
)


def load_marker_positions():
    try:
        with open("./params/calibrated_marker_positions.json", "r") as f:
            pos_json = json.load(f)
            return {int(k): np.array(v) for k, v in pos_json.items()}
    except:
        print("Couldn't load calibrated marker positions, using ideal positions")
    return idealMarkerPositions


def get_focus_target(dist_to_camera):
    f = np.interp([dist_to_camera], focus_targets[:, 0], focus_targets[:, 1])[0]
    return 5 * round(f / 5)  # Webcam only supports multiples of 5


def run_tracker(
    on_estimate: Optional[Callable[[np.ndarray, np.ndarray], None]],
    recording_enabled: Optional[mp.Value] = None,
    recording_timestamp: str = "",
):
    cv2.namedWindow("Tracker", cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Tracker", 1050, int(1050 * 1080 / 1920))
    camera_matrix, dist_coeffs = read_camera_parameters(
        "params/camera_params_c922_f30.yml"
    )
    marker_positions = load_marker_positions()
    print("Opening webcam..")
    webcam = get_webcam()

    calibrated = False
    baseRvec = np.zeros([3, 1])
    baseTvec = np.zeros([3, 1])
    avg_fps = 30

    tracker = MarkerTracker(camera_matrix, dist_coeffs, marker_positions)
    frame_count = 0
    auto_focus = True
    current_focus = 30
    while True:
        frame_start_time = time.perf_counter()
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
        elif keypress == ord("u"):
            auto_focus = False
            current_focus += 5
            webcam.set(cv2.CAP_PROP_FOCUS, current_focus)
        elif keypress == ord("d"):
            auto_focus = False
            current_focus -= 5
            webcam.set(cv2.CAP_PROP_FOCUS, current_focus)
        elif keypress == ord("a"):
            auto_focus = True

        frame: np.ndarray
        ret, frame = webcam.read()
        frame_original = frame.copy()

        if keypress == ord("s"):
            focus = round(webcam.get(cv2.CAP_PROP_FOCUS))
            filepath = f"calibration_pics/f{focus}/{round(time.time())}.jpg"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            success = cv2.imwrite(filepath, frame)
            print(f"save: {success}, {filepath}")

        processing_start_time = time.perf_counter()

        if not calibrated or keypress == ord("c"):
            print("Calibrating...")
            try:
                baseRvec, baseTvec = estimate_camera_pose_charuco(
                    frame, camera_matrix, dist_coeffs
                )
                calibrated = True
            except Exception as e:
                print("Error calibrating camera, press C to retry.", e)

        result = tracker.process_frame(frame)
        processing_end_time = time.perf_counter()
        if result is not None:
            rvec, tvec = result
            rvec_relative, tvec_relative = relative_transform(
                rvec, tvec, baseRvec, baseTvec
            )
            tip_to_imu_offset = -np.array(IMU_OFFSET) - [0, STYLUS_LENGTH, 0]
            rvec_tip, tvec_tip, *_ = cv2.composeRT(
                np.zeros(3), tip_to_imu_offset, rvec, tvec
            )
            _, tvec_tip_relative = relative_transform(
                rvec_tip, tvec_tip, baseRvec, baseTvec
            )
            R_relative = cv2.Rodrigues(rvec_relative)[0]  # TODO: use Rodrigues directly
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.01)
            cv2.drawFrameAxes(
                frame, camera_matrix, dist_coeffs, rvec_tip, tvec_tip, 0.01
            )
            cv2.putText(
                frame,
                f"IMU: [{array_to_str(tvec_relative*100)}]cm",
                (10, 120),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                TEXT_COL,
            )
            cv2.putText(
                frame,
                f"Tip: [{array_to_str(tvec_tip_relative*100)}]cm",
                (10, 150),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                TEXT_COL,
            )

            if on_estimate is not None:
                on_estimate(CameraReading(tvec_relative, R_relative))

        frame_end_time = time.perf_counter()
        fps = 1 / (frame_end_time - frame_start_time)
        avg_fps = 0.9 * avg_fps + 0.1 * fps
        cv2.putText(
            frame,
            f"FPS: {avg_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            TEXT_COL,
        )
        cv2.putText(
            frame,
            f"Processing: {(processing_end_time - processing_start_time)*1000:.1f}ms",
            (10, 60),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            TEXT_COL,
        )
        cv2.putText(
            frame,
            f"Focus: {current_focus}",
            (10, 180),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            TEXT_COL,
        )

        cv2.imshow("Tracker", frame)

        if recording_enabled and recording_enabled.value:
            timestamp = time.time_ns() // 1_000_000
            dir = f"recordings/{recording_timestamp}/frames"
            Path(dir).mkdir(parents=True, exist_ok=True)
            filepath = f"{dir}/{timestamp}.bmp"
            cv2.imwrite(filepath, frame_original)
            with open(
                f"./recordings/{recording_timestamp}/camera_extrinsics.pkl", "wb"
            ) as pickle_file:
                pickle.dump((baseRvec, baseTvec), pickle_file)

        # Adjust focus periodically
        if auto_focus and calibrated and frame_count % focus_interval == 0:
            if result is None:
                focus = 30
            else:
                dist_to_camera = np.linalg.norm(tvec)
                focus = get_focus_target(dist_to_camera)
            if focus != current_focus:
                current_focus = focus
                webcam.set(cv2.CAP_PROP_FOCUS, focus)

        frame_count += 1


if __name__ == "__main__" and sys.flags.interactive == 0:
    run_tracker(None)
