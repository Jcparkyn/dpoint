import time
from typing import NamedTuple

import numpy as np
import cv2 as cv
from scipy.spatial import KDTree
from pyquaternion import Quaternion

from app.filter import DpointFilter, blend_new_data
from app.marker_tracker import CameraReading
from app.monitor_ble import StylusReading
from app.dimensions import IMU_OFFSET, STYLUS_LENGTH

INCH_TO_METRE = 0.0254


def binarize(image: np.ndarray) -> np.ndarray:
    ret, threshold = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return threshold


def reject_outliers_2d(x: np.ndarray, y: np.ndarray, m=2.0):
    d = np.sqrt((x - np.median(x)) ** 2 + (y - np.median(y)) ** 2)
    mdev = np.median(d)
    s = d / mdev if mdev else np.zeros(len(d))
    indices = s < m
    return x[indices], y[indices]


def get_black_points(image: np.ndarray, dpi: float):
    points_0, points_1 = (binarize(image) == 0).nonzero()
    points_x = points_1 * (INCH_TO_METRE / dpi)
    points_y = (image.shape[0] - points_0) * (INCH_TO_METRE / dpi)
    points_x, points_y = reject_outliers_2d(points_x, points_y)
    return points_x, points_y


def normalize_points(xy: np.ndarray):
    return xy - np.mean(xy, axis=0)


def camera_reading_to_tip_pos(reading: CameraReading):
    orientation_quat = Quaternion(matrix=reading.orientation_mat)
    tip_pos = reading.position.flatten() - orientation_quat.rotate(
        np.array([0, STYLUS_LENGTH, 0]) + IMU_OFFSET
    ).flatten()
    return tip_pos


def replay_data(recorded_data: list[tuple[float, CameraReading | StylusReading]], dt, smoothing_length, camera_delay):
    filter = DpointFilter(dt=dt, smoothing_length=smoothing_length, camera_delay=camera_delay)
    sample_count = sum(
        isinstance(reading, StylusReading) for _, reading in recorded_data
    )
    print(f"sample_count: {sample_count}")

    tip_pos_predicted = np.zeros((sample_count, 3))
    tip_pos_smoothed = np.zeros((sample_count, 3))
    pressure = np.zeros(sample_count)

    tip_pos_cameraonly = []
    pressure_cameraonly = []

    camera_fuse_times = []
    stylus_fuse_times = []

    pressure_baseline = 0.017  # Approximate measured value for initial estimate
    pressure_avg_factor = 0.1  # Factor for exponential moving average
    pressure_range = 0.02
    pressure_offset = 0.003  # Offset so that small positive numbers are treated as zero
    sample = 0
    for t, reading in recorded_data:
        t0 = time.perf_counter()
        match reading:
            case CameraReading(pos, or_mat):
                # print(f"t: {t}, pos: {pos}, or_mat: {or_mat}")
                tip_pos_cameraonly.append(camera_reading_to_tip_pos(reading))
                pressure_cameraonly.append(pressure[sample - 4])
                smoothed_tip_pos = filter.update_camera(pos.flatten(), or_mat)
                if smoothed_tip_pos:
                    start = sample - len(smoothed_tip_pos) + 1
                    tps_view = tip_pos_smoothed[start : sample + 1, :]
                    tps_view[:,:] = blend_new_data(tps_view, smoothed_tip_pos, 0.5)
                camera_fuse_times.append(time.perf_counter() - t0)
            case StylusReading(accel=accel, gyro=gyro, t=_, pressure=p):
                filter.update_imu(accel, gyro)
                position, orientation = filter.get_tip_pose()
                zpos = position[2]
                if zpos > 0.005:
                    # calibrate pressure baseline using current pressure reading
                    pressure_baseline = (
                        pressure_baseline * (1 - pressure_avg_factor)
                        + reading.pressure * pressure_avg_factor
                    )
                tip_pos_predicted[sample, :] = position.flatten()
                tip_pos_smoothed[sample, :] = position.flatten()
                pressure[sample] = (
                    p - pressure_baseline - pressure_offset
                ) / pressure_range
                stylus_fuse_times.append(time.perf_counter() - t0)
                sample += 1
            case _:
                print("Invalid reading", reading)
    camera_fuse_times = np.array(camera_fuse_times)*1000
    stylus_fuse_times = np.array(stylus_fuse_times)*1000
    print(f"Camera: {np.mean(camera_fuse_times):.3f}ms +- {np.std(camera_fuse_times):.3f}")
    print(f"Stylus: {np.mean(stylus_fuse_times):.3f}ms +- {np.std(stylus_fuse_times):.3f}")
    return tip_pos_predicted, tip_pos_smoothed, pressure, np.row_stack(tip_pos_cameraonly), np.array(pressure_cameraonly)


def minimise_chamfer_distance(a: np.ndarray, b: np.ndarray, iterations=3):
    """Finds the optimal translation of b to minimise the chamfer distance to a."""
    tree = KDTree(a)
    offset = np.mean(a, axis=0) - np.mean(b, axis=0)
    for i in range(iterations):
        _, indices = tree.query(b + offset)
        errors = (a[indices, :] - b) - offset
        error = np.mean(errors, axis=0)
        offset += error
        # print(f"Iteration {i}, offset: {offset}, error: {error}")
    dist, _ = tree.query(b + offset)
    return offset, dist


def resample_line(points, desired_distance, mask):
    assert points.shape[1] == 2
    if points.shape[0] < 2:
        raise ValueError("The input array should contain at least 2 points.")

    # Calculate the total length of the line
    lengths = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    total_length = np.sum(lengths)

    # Calculate the number of new points to be added
    num_points = int(np.ceil(total_length / desired_distance))

    # Calculate the distances between the original points
    distances = np.zeros(len(points))
    distances[1:] = np.cumsum(lengths)

    # Interpolate new points along the line
    new_distances = np.linspace(0, total_length, num_points)
    resampled_points = np.zeros((num_points, 2))
    resampled_mask = np.interp(new_distances, distances, mask)
    for i in range(2):
        resampled_points[:, i] = np.interp(new_distances, distances, points[:, i])

    return resampled_points[resampled_mask > 0.5, :]


def merge_data(
    stylus_data: list, camera_data: list
) -> list[tuple[float, CameraReading | StylusReading]]:
    result = stylus_data + camera_data
    result.sort(key=lambda x: x[0])
    return result


class ProcessedStroke(NamedTuple):
    position: np.ndarray
    pressure: np.ndarray
    dist_mean: float


class ProcessResult(NamedTuple):
    pressure: np.ndarray
    paths: dict[str, ProcessedStroke]


def process_stroke(
    stroke: np.ndarray, scan_points: np.ndarray, pressure: np.ndarray
) -> ProcessedStroke:
    resample_dist = 0.001 * 0.5  # 0.5mm
    stroke_resampled = resample_line(
        stroke[:, :2], resample_dist, mask=pressure > 0.1
    )
    offset, dist = minimise_chamfer_distance(
        scan_points, stroke_resampled, iterations=8
    )
    offset3d = np.append(offset, 0)
    dist_mean = np.mean(dist)
    return ProcessedStroke(
        position=stroke + offset3d, pressure=pressure, dist_mean=dist_mean
    )
