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


def readCameraParameters(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise Exception("Couldn't open file")
    cameraMatrix = fs.getNode("camera_matrix").mat()
    distCoeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return (cameraMatrix, distCoeffs)


def getWebcam():
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    webcam.set(cv2.CAP_PROP_FPS, 60)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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


def inverseRT(rvec, tvec) -> Tuple[np.ndarray, np.ndarray]:
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.transpose(R)
    return (cv2.Rodrigues(Rt)[0], -Rt @ tvec)


def relativeTransform(rvec1, tvec1, rvec2, tvec2) -> Tuple[np.ndarray, np.ndarray]:
    rvec2inv, tvec2inv = inverseRT(rvec2, tvec2)
    rvec, tvec, *_ = cv2.composeRT(rvec1, tvec1, rvec2inv, tvec2inv)
    return (rvec, tvec)


def getArucoParams():
    arucoParams = aruco.DetectorParameters()
    arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    arucoParams.cornerRefinementWinSize = 2
    # Reduce the number of threshold steps, which significantly improves performance
    arucoParams.adaptiveThreshWinSizeMin = 15
    arucoParams.adaptiveThreshWinSizeMax = 15
    arucoParams.useAruco3Detection = False
    arucoParams.minMarkerPerimeterRate = 0.02
    arucoParams.maxMarkerPerimeterRate = 0.5
    arucoParams.minSideLengthCanonicalImg = 16
    arucoParams.adaptiveThreshConstant = 7
    return arucoParams


arucoDic = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
arucoParams = getArucoParams()
detector = aruco.ArucoDetector(arucoDic, arucoParams)

reprojectionErrorThreshold = 3  # px


def array_to_str(arr):
    return ",".join(map(lambda x: f"{x:+2.2f}", list(arr.flat)))


charuco_dic = aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
charuco_board = aruco.CharucoBoard((10, 7), 0.028, 0.022, charuco_dic)
charuco_params = aruco.DetectorParameters()
charuco_detector = aruco.ArucoDetector(charuco_dic, charuco_params)


def estimate_camera_pose_charuco(frame, cameraMatrix, distCoeffs):
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
        cameraMatrix,
        distCoeffs,
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
        display_frame, cameraMatrix, distCoeffs, rvec, tvec, 0.2
    )
    # cv2.imshow("Charuco", display_frame)
    return (rvec, tvec)


def vector_rms(arr: np.ndarray, axis: int):
    """Computes the RMS magnitude of an array of vectors."""
    return math.sqrt(np.mean(np.sum(np.square(arr), axis=axis)))


def solve_pnp(
    initialized, prevRvec, prevTvec, objectPoints, imagePoints, cameraMatrix, distCoeffs
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Attempt to refine the previous pose. If this fails, fall back to SQPnP."""
    if initialized:
        rvec, tvec = cv2.solvePnPRefineVVS(
            objectPoints,
            imagePoints,
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoeffs,
            # OpenCV mutates these arguments, which we don't want.
            rvec=prevRvec.copy(),
            tvec=prevTvec.copy(),
        )
        projectedImagePoints, _ = cv2.projectPoints(
            objectPoints, rvec, tvec, cameraMatrix, distCoeffs, None
        )
        projectedImagePoints = projectedImagePoints[:, 0, :]
        reprojectionError = vector_rms(projectedImagePoints - imagePoints, axis=1)

        if reprojectionError < reprojectionErrorThreshold:
            # print(f"Reprojection error: {reprojectionError}")
            return (True, rvec, tvec)
        else:
            print(f"Reprojection error too high: {reprojectionError}")

    success, rvec, tvec = cv2.solvePnP(
        objectPoints,
        imagePoints,
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    return (success, rvec, tvec)


MarkerDict = dict[int, tuple[np.ndarray, np.ndarray]]


def detect_markers_bounded(
    detector, frame: np.ndarray, x0: int, x1: int, y0: int, y1: int
):
    x0, y0 = max(x0, 0), max(y0, 0)
    frame_view = frame[y0:y1, x0:x1]
    allCornersIS, ids, rejected = detector.detectMarkers(frame_view)
    if ids is not None:
        for i in range(ids.shape[0]):
            allCornersIS[i][0, :, 0] += x0
            allCornersIS[i][0, :, 1] += y0
    return allCornersIS, ids, rejected


class MarkerTracker:
    def __init__(
        self,
        cameraMatrix: np.ndarray,
        distCoeffs: np.ndarray,
        markerPositions: dict[int, np.ndarray],
    ):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None
        self.initialized = False
        self.markerPositions = markerPositions
        self.lastValidMarkers: MarkerDict = {}
        self.next_search_area: Optional[tuple[int]] = None

    @staticmethod
    def get_search_area(velocity: np.array, validMarkers: MarkerDict):
        """Returns a bounding box to search in the next frame, based on the current marker positions and velocity."""
        all_corners = np.concatenate(
            [cornersIS for _, cornersIS in validMarkers.values()]
        )

        def minmax(x):
            return np.min(x), np.max(x)

        x0, x1 = minmax(all_corners[:, 0] + velocity[0] / FPS)
        y0, y1 = minmax(all_corners[:, 1] + velocity[1] / FPS)
        w = x1 - x0
        h = y1 - y0
        # Amount to expand each axis by, in pixels. This is just a rough heuristic, and the constants are arbitrary.
        expand = 0.7 * max(w + h, 350) + 1.0 * velocity / FPS
        return (
            int(x0 - expand[0]),
            int(x1 + expand[0]),
            int(y0 - expand[1]),
            int(y1 + expand[1]),
        )

    def process_frame(self, frame: np.ndarray):
        ids: np.ndarray
        if self.next_search_area is None:
            allCornersIS, ids, rejected = detector.detectMarkers(frame)
        else:
            x0, x1, y0, y1 = self.next_search_area
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 100, 0, 0.3), 2)
            allCornersIS, ids, rejected = detect_markers_bounded(
                detector, frame, x0, x1, y0, y1
            )
        aruco.drawDetectedMarkers(frame, allCornersIS, ids)
        validMarkers: MarkerDict = {}
        if ids is not None:
            for i in range(ids.shape[0]):
                # cornersIS is 4x2
                id, cornersIS = (ids[i, 0], allCornersIS[i][0, :, :])
                if id in self.markerPositions:
                    cornersPS = self.markerPositions[id]
                    validMarkers[id] = (cornersPS, cornersIS)

        if len(validMarkers) < 1:
            self.initialized = False
            self.lastValidMarkers = {}
            self.next_search_area = None
            return None

        pointDeltas = []
        for id, (cornersPS, cornersIS) in validMarkers.items():
            if id in self.lastValidMarkers:
                velocity = cornersIS - self.lastValidMarkers[id][1]
                pointDeltas.append(np.mean(velocity, axis=0))

        if pointDeltas:
            meanVelocity = np.mean(pointDeltas, axis=0) * 30  # px/second
        else:
            meanVelocity = np.zeros(2)

        meanPositionIS = np.mean(
            [cornersIS for _, cornersIS in validMarkers.values()],
            axis=(0, 1),
        )

        self.next_search_area = self.get_search_area(meanVelocity, validMarkers)

        screenCorners = []
        penCorners = []
        delayPerImageRow = 1 / 30 / 1080  # seconds/row

        for id, (cornersPS, cornersIS) in validMarkers.items():
            penCorners.append(cornersPS)
            if pointDeltas:
                # Compensate for rolling shutter
                timeDelay = (
                    cornersIS[:, 1] - meanPositionIS[1]
                ) * delayPerImageRow  # seconds, relative to centroid
                cornersISCompensated = (
                    cornersIS - meanVelocity * timeDelay[:, np.newaxis]
                )
                screenCorners.append(cornersISCompensated)
            else:
                screenCorners.append(cornersIS)

        self.initialized, self.rvec, self.tvec = solve_pnp(
            self.initialized,
            self.rvec,
            self.tvec,
            objectPoints=np.concatenate(penCorners),
            imagePoints=np.concatenate(screenCorners),
            cameraMatrix=self.cameraMatrix,
            distCoeffs=self.distCoeffs,
        )

        self.lastValidMarkers = validMarkers
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
    cv2.moveWindow("Tracker", -1080, -120)
    cv2.resizeWindow("Tracker", 1050, int(1050 * 1080 / 1920))
    cameraMatrix, distCoeffs = readCameraParameters("params/camera_params_c922_f30.yml")
    markerPositions = load_marker_positions()
    print("Opening webcam..")
    webcam = getWebcam()

    calibrated = False
    baseRvec = np.zeros([3, 1])
    baseTvec = np.zeros([3, 1])
    avg_fps = 30

    tracker = MarkerTracker(cameraMatrix, distCoeffs, markerPositions)
    frame_count = 0
    auto_focus = True
    current_focus = 30
    while True:
        frameStartTime = time.perf_counter()
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

        processingStartTime = time.perf_counter()

        if not calibrated or keypress == ord("c"):
            calibrated = True
            print("Calibrating...")
            baseRvec, baseTvec = estimate_camera_pose_charuco(
                frame, cameraMatrix, distCoeffs
            )

        result = tracker.process_frame(frame)
        processingEndTime = time.perf_counter()
        if result is not None:
            rvec, tvec = result
            rvecRelative, tvecRelative = relativeTransform(
                rvec, tvec, baseRvec, baseTvec
            )
            tip_to_imu_offset = -np.array(IMU_OFFSET) - [0, STYLUS_LENGTH, 0]
            rvecTip, tvecTip, *_ = cv2.composeRT(
                np.zeros(3), tip_to_imu_offset, rvec, tvec
            )
            _, tvecTipRelative = relativeTransform(rvecTip, tvecTip, baseRvec, baseTvec)
            Rrelative = cv2.Rodrigues(rvecRelative)[0]  # TODO: use Rodrigues directly
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.01)
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecTip, tvecTip, 0.01)
            cv2.putText(
                frame,
                f"IMU: [{array_to_str(tvecRelative*100)}]cm",
                (10, 120),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (255, 0, 0),
            )
            cv2.putText(
                frame,
                f"Tip: [{array_to_str(tvecTipRelative*100)}]cm",
                (10, 150),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (255, 0, 0),
            )

            if on_estimate is not None:
                on_estimate(CameraReading(tvecRelative, Rrelative))

        frameEndTime = time.perf_counter()
        fps = 1 / (frameEndTime - frameStartTime)
        avg_fps = 0.9 * avg_fps + 0.1 * fps
        cv2.putText(
            frame,
            f"FPS: {avg_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 255, 0),
        )
        cv2.putText(
            frame,
            f"Processing: {(processingEndTime - processingStartTime)*1000:.1f}ms",
            (10, 60),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 255, 0),
        )
        cv2.putText(
            frame,
            f"Focus: {current_focus}",
            (10, 180),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (255, 0, 0),
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
        if auto_focus and frame_count % focus_interval == 0:
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
