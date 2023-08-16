import pickle

import numpy as np
from app import get_line_color_from_pressure

from filter import DpointFilter
from marker_tracker import CameraReading
from monitor_ble import StylusReading
from vispy import plot as vp
from vispy.color import Color
from vispy.plot import PlotWidget
from PIL import Image
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


def replay_data(recorded_data: list[tuple[float, CameraReading | StylusReading]]):
    filter = DpointFilter(dt=1 / 120)
    sample_count = sum(
        isinstance(reading, StylusReading) for _, reading in recorded_data
    )
    print(f"sample_count: {sample_count}")

    tip_pos_predicted = np.zeros((sample_count, 3))
    tip_pos_smoothed = np.zeros((sample_count, 3))
    pressure = np.zeros(sample_count)

    pressure_baseline = 0.017  # Approximate measured value for initial estimate
    pressure_range = 0.02
    pressure_offset = 0.002  # Offset so that small positive numbers are treated as zero
    sample = 0
    for t, reading in recorded_data:
        match reading:
            case CameraReading(pos, or_mat):
                # print(f"t: {t}, pos: {pos}, or_mat: {or_mat}")
                smoothed_tip_pos = filter.update_camera(pos.flatten(), or_mat)
                if smoothed_tip_pos:
                    start = sample - len(smoothed_tip_pos) + 1
                    tip_pos_smoothed[start : sample + 1, :] = smoothed_tip_pos
            case StylusReading(accel, gyro, _, p):
                # print(f"t: {t}, accel: {accel}, gyro: {gyro}, pressure: {p}")
                filter.update_imu(accel, gyro)
                position, orientation = filter.get_tip_pose()
                tip_pos_predicted[sample, :] = position.flatten()
                pressure[sample] = (
                    p - pressure_baseline - pressure_offset
                ) / pressure_range
                sample += 1
    return tip_pos_predicted, tip_pos_smoothed, pressure


def chamfer_distance(a: np.ndarray, b: np.ndarray):
    tree = KDTree(a)
    dist, indices = tree.query(b)
    return dist


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
    return offset


def resample_line(points, pressure, desired_distance):
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
    resampled_pressure = np.interp(new_distances, distances, pressure)
    for i in range(2):
        resampled_points[:, i] = np.interp(new_distances, distances, points[:, i])

    return resampled_points, resampled_pressure


if __name__ == "__main__":
    recorded_data: list[tuple[float, CameraReading | StylusReading]] = []
    with open("./recordings/20230816_170100/recorded_data.pickle", "rb") as pickle_file:
        recorded_data = pickle.load(pickle_file)

    tip_pos_predicted, tip_pos_smoothed, pressure = replay_data(recorded_data)
    s0, s1 = 100, -100
    tip_pos_predicted = tip_pos_predicted[s0:s1, :]
    tip_pos_smoothed = tip_pos_smoothed[s0:s1, :]
    pressure = pressure[s0:s1]

    fig = vp.Fig(size=(800, 800), show=False)
    ax: PlotWidget = fig[0, 0]

    scan_img = Image.open("./recordings/20230816_170100/scan.jpg")
    scan_dpi, _ = scan_img.info.get("dpi")
    scan_x, scan_y = get_black_points(np.asarray(scan_img), scan_dpi)
    scan_points = normalize_points(np.column_stack((scan_x, scan_y)))

    tps_resampled, pressure_resampled = resample_line(
        tip_pos_smoothed[:, :2], pressure, 0.001 * 0.5
    )
    tps_black = tps_resampled[pressure_resampled > 0.1, :]
    offset = minimise_chamfer_distance(scan_points, tps_black, iterations=3)
    offset3d = np.append(offset, 0)

    ax.plot(
        tip_pos_predicted + offset3d,
        marker_size=0,
        color=get_line_color_from_pressure(pressure, (0.7, 0.2, 0.2)),
    )
    ax.plot(
        tip_pos_smoothed + offset3d,
        marker_size=0,
        width=2,
        color=get_line_color_from_pressure(pressure, (0, 0, 0)),
    )
    ax.view.camera.aspect = 1

    ax.plot(scan_points, marker_size=1, width=0, edge_color=(0, 0, 1, 0.2))
    # ax.plot(tps_black + offset, marker_size=1, color=(0, 1, 0), width=0)

    dist = chamfer_distance(scan_points, tps_black + offset)
    print(f"Chamfer distance: {np.mean(dist)*1000:0.4f}mm")

    fig.show(run=True)
