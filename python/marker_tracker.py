import cv2
from cv2 import aruco
import numpy as np
from typing import Tuple, Callable, Optional
import time
import sys


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


markerLength = 0.015
markerCorners = np.array(
    [
        [-markerLength / 2, markerLength / 2, 0],
        [markerLength / 2, markerLength / 2, 0],
        [markerLength / 2, -markerLength / 2, 0],
        [-markerLength / 2, -markerLength / 2, 0],
    ],
    dtype=np.float32,
)


def getCornersPS(origin: np.ndarray, angleY: float) -> np.ndarray:
    cornersWS = markerCorners + origin
    rotated_corners = np.apply_along_axis(lambda x: rotateY(angleY, x), 1, cornersWS)
    return rotated_corners


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
    0: getCornersPS(np.array([0, 0.147, 0.01], dtype=np.float32), deg2rad(45)),
    1: getCornersPS(np.array([0, 0.147, 0.01], dtype=np.float32), deg2rad(135)),
    2: getCornersPS(np.array([0, 0.147, 0.01], dtype=np.float32), deg2rad(225)),
    3: getCornersPS(np.array([0, 0.147, 0.01], dtype=np.float32), deg2rad(315)),
    4: getCornersPS(np.array([0, 0.1175, 0.011], dtype=np.float32), deg2rad(90)),
    5: getCornersPS(np.array([0, 0.1175, 0.011], dtype=np.float32), deg2rad(180)),
    6: getCornersPS(np.array([0, 0.1170, 0.011], dtype=np.float32), deg2rad(270)),
    7: getCornersPS(np.array([0, 0.1175, 0.011], dtype=np.float32), deg2rad(5)),
}

baseMarkerId = 40
baseMarkerLength = 0.1
baseMarkerCornersWS = np.array(
    [
        [-baseMarkerLength / 2, baseMarkerLength / 2, 0],
        [baseMarkerLength / 2, baseMarkerLength / 2, 0],
        [baseMarkerLength / 2, -baseMarkerLength / 2, 0],
        [-baseMarkerLength / 2, -baseMarkerLength / 2, 0],
    ],
    dtype=np.float32,
)

arucoDic = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoParams = aruco.DetectorParameters()
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
arucoParams.cornerRefinementWinSize = 2
# arucoParams.cornerRefinementMaxIterations = 30
# arucoParams.cornerRefinementMinAccuracy = 0.01
# arucoParams.errorCorrectionRate = 0.99
# arucoParams.maxErroneousBitsInBorderRate = 0.99
# Reduce the number of threshold steps, which significantly improves performance
arucoParams.adaptiveThreshWinSizeMin = 23
arucoParams.adaptiveThreshWinSizeMax = 23
# arucoParams.adaptiveThreshWinSizeStep = 1000
arucoParams.useAruco3Detection = False
arucoParams.minMarkerPerimeterRate = 0.02
arucoParams.maxMarkerPerimeterRate = 1
arucoParams.minSideLengthCanonicalImg = 16
detector = aruco.ArucoDetector(arucoDic, arucoParams)


def array_to_str(arr):
    return ",".join(map(lambda x: f"{x*100:+2.1f}", list(arr.flat)))


def run_tracker(on_estimate: Optional[Callable[[np.ndarray, np.ndarray], None]]):
    cv2.namedWindow("Tracker", cv2.WINDOW_KEEPRATIO)
    cv2.moveWindow("Tracker", -1080, -150)
    cameraMatrix, distCoeffs = readCameraParameters("camera_params_c922_f30.yml")
    print("Opening webcam..")
    webcam = getWebcam()

    rvec, tvec = (None, None)
    initialized = False
    calibrated = False
    baseRvec = np.zeros([3, 1])
    baseTvec = np.zeros([3, 1])
    avg_fps = 30

    while True:
        frameStartTime = time.perf_counter()
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
        elif keypress == ord("u"):
            webcam.set(cv2.CAP_PROP_FOCUS, webcam.get(cv2.CAP_PROP_FOCUS) + 5)
        elif keypress == ord("d"):
            webcam.set(cv2.CAP_PROP_FOCUS, webcam.get(cv2.CAP_PROP_FOCUS) - 5)

        ret, frame = webcam.read()

        if keypress == ord("s"):
            focus = round(webcam.get(cv2.CAP_PROP_FOCUS))
            filepath = f"calibration_pics/f{focus}/{round(time.time())}.jpg"
            success = cv2.imwrite(filepath, frame)
            # success = cv2.imwrite(f"calibration_pics/f30/1234.png", frame)
            print(f"save: {success}, {filepath}")

        processingStartTime = time.perf_counter()
        cv2.flip(frame, -1, frame)

        ids: np.ndarray
        allCornersIS: np.ndarray
        allCornersIS, ids, rejected = detector.detectMarkers(frame)
        aruco.drawDetectedMarkers(frame, allCornersIS, ids)

        validMarkers = []
        if ids is not None:
            for i in range(ids.shape[0]):
                id, cornersIS = (ids[i, 0], allCornersIS[i][0, :, :])
                if id in markersOnPen:
                    cornersPS = markersOnPen[id]
                    validMarkers.append((id, cornersPS, cornersIS))

            if baseMarkerId in ids and (not calibrated or keypress == ord("c")):
                calibrated = True
                print("Calibrating...")
                baseMarkerIndex = list(ids).index(baseMarkerId)
                baseMarkerCornersIS = allCornersIS[baseMarkerIndex]
                success, baseRvec, baseTvec = cv2.solvePnP(
                    baseMarkerCornersWS, baseMarkerCornersIS, cameraMatrix, distCoeffs
                )

        if len(validMarkers) >= 1:
            screenCorners = np.concatenate(
                [cornersIS for _, _, cornersIS in validMarkers]
            )
            penCorners = np.concatenate([cornersPS for _, cornersPS, _ in validMarkers])
            success, rvec, tvec = cv2.solvePnP(
                penCorners,
                screenCorners,
                cameraMatrix=cameraMatrix,
                distCoeffs=distCoeffs,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=initialized,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            rvecRelative, tvecRelative = relativeTransform(
                rvec, tvec, baseRvec, baseTvec
            )
            Rrelative = cv2.Rodrigues(rvecRelative)[0]  # TODO: use Rodrigues directly
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, markerLength)
            cv2.putText(
                frame,
                f"Tip: [{array_to_str(tvecRelative)}]",
                (10, 120),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0),
            )
            if on_estimate is not None:
                on_estimate(Rrelative, tvecRelative)
            initialized = success
        else:
            initialized = False

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
