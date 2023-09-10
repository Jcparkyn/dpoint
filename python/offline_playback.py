from typing import NamedTuple

import numpy as np

from filter import DpointFilter, blend_new_data
from marker_tracker import CameraReading
from monitor_ble import StylusReading
import cv2 as cv
from scipy.spatial import KDTree

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


def replay_data(recorded_data: list[tuple[float, CameraReading | StylusReading]], smoothing_length):
    filter = DpointFilter(dt=1 / 120, smoothing_length=smoothing_length)
    sample_count = sum(
        isinstance(reading, StylusReading) for _, reading in recorded_data
    )
    print(f"sample_count: {sample_count}")

    tip_pos_predicted = np.zeros((sample_count, 3))
    tip_pos_smoothed = np.zeros((sample_count, 3))
    pressure = np.zeros(sample_count)

    pressure_baseline = 0.017  # Approximate measured value for initial estimate
    pressure_avg_factor = 0.1  # Factor for exponential moving average
    pressure_range = 0.02
    pressure_offset = 0.003  # Offset so that small positive numbers are treated as zero
    sample = 0
    for t, reading in recorded_data:
        match reading:
            case CameraReading(pos, or_mat):
                # print(f"t: {t}, pos: {pos}, or_mat: {or_mat}")
                smoothed_tip_pos = filter.update_camera(pos.flatten(), or_mat)
                if smoothed_tip_pos:
                    start = sample - len(smoothed_tip_pos) + 1
                    tps_view = tip_pos_smoothed[start : sample + 1, :]
                    tps_view[:,:] = blend_new_data(tps_view, smoothed_tip_pos, 0.5)
            case StylusReading(accel, gyro, _, p):
                # print(f"t: {t}, accel: {accel}, gyro: {gyro}, pressure: {p}")
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
                sample += 1
    return tip_pos_predicted, tip_pos_smoothed, pressure


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
    result = [x for x in stylus_data if isinstance(x[1], StylusReading)] + camera_data
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
    s0, s1 = 70, -70
    resample_dist = 0.001 * 0.5  # 0.5mm
    stroke2 = stroke[s0:s1, :]
    stroke_resampled = resample_line(
        stroke2[:, :2], resample_dist, mask=pressure[s0:s1] > 0.1
    )
    offset, dist = minimise_chamfer_distance(
        scan_points, stroke_resampled, iterations=8
    )
    offset3d = np.append(offset, 0)
    dist_mean = np.mean(dist)
    return ProcessedStroke(
        position=stroke2 + offset3d, pressure=pressure[s0:s1], dist_mean=dist_mean
    )
