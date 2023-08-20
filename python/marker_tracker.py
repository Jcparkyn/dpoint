import math
import pickle
import cv2
from cv2 import aruco
import numpy as np
from typing import NamedTuple, Tuple, Callable, Optional
import time
import sys

from dimensions import IMU_OFFSET, STYLUS_LENGTH


class CameraReading(NamedTuple):
    position: np.ndarray
    orientation_mat: np.ndarray


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
    webcam.set(cv2.CAP_PROP_EXPOSURE, -8)
    webcam.set(cv2.CAP_PROP_BRIGHTNESS, 127)
    webcam.set(cv2.CAP_PROP_CONTRAST, 140)
    webcam.set(cv2.CAP_PROP_GAIN, 200)
    webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return webcam


def rotateY(angle: float, point: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return np.dot(rotation_matrix, point)


# markerLength = 0.015
def getMarkerCorners(markerLength: float):
    return np.array(
        [
            [-markerLength / 2, markerLength / 2, 0],
            [markerLength / 2, markerLength / 2, 0],
            [markerLength / 2, -markerLength / 2, 0],
            [-markerLength / 2, -markerLength / 2, 0],
        ],
        dtype=np.float32,
    )


def getCornersPS(
    origin: np.ndarray, angleY: float, markerLength: float = 0.013*0.97
) -> np.ndarray:
    cornersWS = getMarkerCorners(markerLength) + origin
    rotated_corners = np.apply_along_axis(lambda x: rotateY(angleY, x), 1, cornersWS)
    return rotated_corners - IMU_OFFSET


def deg2rad(deg: float) -> float:
    return deg * np.pi / 180


def inverseRT(rvec, tvec) -> Tuple[np.ndarray, np.ndarray]:
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.transpose(R)
    return (cv2.Rodrigues(Rt)[0], -Rt @ tvec)


def relativeTransform(rvec1, tvec1, rvec2, tvec2) -> Tuple[np.ndarray, np.ndarray]:
    rvec2inv, tvec2inv = inverseRT(rvec2, tvec2)
    rvec, tvec, *_ = cv2.composeRT(rvec1, tvec1, rvec2inv, tvec2inv)
    return (rvec, tvec)


markersOnPen = {
    99: getCornersPS(np.array([0, -0.01, 0.01], dtype=np.float32), deg2rad(135)),
    98: getCornersPS(np.array([0, -0.01, 0.01], dtype=np.float32), deg2rad(225)),
    97: getCornersPS(np.array([0, -0.01, 0.01], dtype=np.float32), deg2rad(315)),
    96: getCornersPS(np.array([0, -0.01, 0.01], dtype=np.float32), deg2rad(45)),
    95: getCornersPS(np.array([0, -0.0393, 0.01], dtype=np.float32), deg2rad(90)),
    94: getCornersPS(np.array([0, -0.0390, 0.01], dtype=np.float32), deg2rad(180)),
    93: getCornersPS(np.array([0, -0.0393, 0.01], dtype=np.float32), deg2rad(270)),
    92: getCornersPS(np.array([0, -0.0390, 0.01], dtype=np.float32), deg2rad(0)),
}

# with open("markersOnPenCalibrated.pkl", "rb") as f:
#     markersOnPenCalibrated = pickle.load(f)

markersOnPenCalibrated = markersOnPen

arucoDic = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
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
detector = aruco.ArucoDetector(arucoDic, arucoParams)

reprojectionErrorThreshold = 3  # px


def array_to_str(arr):
    return ",".join(map(lambda x: f"{x:+2.1f}", list(arr.flat)))


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
            rvec=prevRvec,
            tvec=prevTvec,
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


class MarkerTracker:
    def __init__(self, cameraMatrix: np.ndarray, distCoeffs: np.ndarray):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None
        self.initialized = False
        pass

    def process_frame(self, frame: np.ndarray):
        ids: np.ndarray
        allCornersIS, ids, rejected = detector.detectMarkers(frame)
        aruco.drawDetectedMarkers(frame, allCornersIS, ids)
        validMarkers = []
        if ids is not None:
            for i in range(ids.shape[0]):
                id, cornersIS = (ids[i, 0], allCornersIS[i][0, :, :])
                if id in markersOnPenCalibrated:
                    cornersPS = markersOnPenCalibrated[id]
                    validMarkers.append((id, cornersPS, cornersIS))

        if len(validMarkers) < 1:
            self.initialized = False
            return None

        screenCorners = np.concatenate([cornersIS for _, _, cornersIS in validMarkers])
        penCorners = np.concatenate([cornersPS for _, cornersPS, _ in validMarkers])

        self.initialized, self.rvec, self.tvec = solve_pnp(
            self.initialized,
            self.rvec,
            self.tvec,
            objectPoints=penCorners,
            imagePoints=screenCorners,
            cameraMatrix=self.cameraMatrix,
            distCoeffs=self.distCoeffs,
        )

        return (self.rvec, self.tvec)


def run_tracker(on_estimate: Optional[Callable[[np.ndarray, np.ndarray], None]]):
    cv2.namedWindow("Tracker", cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow("Tracker", -1080, -150)
    cameraMatrix, distCoeffs = readCameraParameters("camera_params_c922_f30.yml")
    print("Opening webcam..")
    webcam = getWebcam()

    calibrated = False
    baseRvec = np.zeros([3, 1])
    baseTvec = np.zeros([3, 1])
    avg_fps = 30

    tracker = MarkerTracker(cameraMatrix, distCoeffs)

    while True:
        frameStartTime = time.perf_counter()
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
        elif keypress == ord("u"):
            webcam.set(cv2.CAP_PROP_FOCUS, webcam.get(cv2.CAP_PROP_FOCUS) + 5)
        elif keypress == ord("d"):
            webcam.set(cv2.CAP_PROP_FOCUS, webcam.get(cv2.CAP_PROP_FOCUS) - 5)

        frame: np.ndarray
        ret, frame = webcam.read()

        if keypress == ord("s"):
            focus = round(webcam.get(cv2.CAP_PROP_FOCUS))
            filepath = f"marker_calibration_pics/f{focus}/{round(time.time())}.jpg"
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
        if result is not None:
            rvec, tvec = result
            rvecRelative, tvecRelative = relativeTransform(
                rvec, tvec, baseRvec, baseTvec
            )
            tip_to_imu_offset = -np.array(IMU_OFFSET) - [0, STYLUS_LENGTH, 0]
            rvecTip, tvecTip, *_ = cv2.composeRT(
                np.zeros(3), tip_to_imu_offset, rvec, tvec
            )
            Rrelative = cv2.Rodrigues(rvecRelative)[0]  # TODO: use Rodrigues directly
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.01)
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecTip, tvecTip, 0.01)
            cv2.putText(
                frame,
                f"IMU: [{array_to_str(tvecRelative*100)}]cm",
                (10, 120),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0),
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
            f"Processing: {(frameEndTime - processingStartTime)*1000:.1f}ms",
            (10, 60),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 255, 0),
        )

        cv2.imshow("Tracker", frame)


if __name__ == "__main__" and sys.flags.interactive == 0:
    run_tracker(None)
