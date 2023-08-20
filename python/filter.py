from collections import deque
from typing import Deque, Tuple
import numpy as np
from numpy import typing as npt
from pyquaternion import Quaternion
from numba.typed.typedlist import List

from dimensions import IMU_OFFSET, STYLUS_LENGTH
from filter_core import (
    STATE_SIZE,
    FilterState,
    HistoryItem,
    SmoothingHistoryItem,
    ekf_predict,
    ekf_smooth,
    fuse_camera,
    fuse_imu,
    i_acc,
    i_accbias,
    i_av,
    i_gyrobias,
    i_pos,
    i_quat,
    i_vel,
)

Mat = npt.NDArray[np.float64]

additive_noise = np.zeros(STATE_SIZE)
additive_noise[i_pos] = 0
additive_noise[i_vel] = 1e-4
additive_noise[i_acc] = 200
additive_noise[i_av] = 50
additive_noise[i_quat] = 2e-4
additive_noise[i_accbias] = 1e-4
additive_noise[i_gyrobias] = 2e-5
Q = np.diag(additive_noise)

accel_noise = 2e-3
gyro_noise = 5e-4
imu_noise = np.diag([accel_noise] * 3 + [gyro_noise] * 3)
camera_noise_pos = 0.1e-5
camera_noise_or = 0.5e-4
camera_noise = np.diag([camera_noise_pos] * 3 + [camera_noise_or] * 4)

smoothing_length = 18


def initial_state(position=None, orientation=None):
    state = np.zeros(STATE_SIZE, dtype=np.float64)
    state[i_quat] = [1, 0, 0, 0]
    if position is not None:
        state[i_pos] = position.flatten()
    if orientation is not None:
        state[i_quat] = orientation.flatten()
    covdiag = np.ones(STATE_SIZE, dtype=np.float64) * 0.0001
    covdiag[i_accbias] = 1e-2
    covdiag[i_gyrobias] = 1e-4
    statecov = np.diag(covdiag)
    return FilterState(state, statecov)


def get_tip_pose(state: Mat) -> Tuple[Mat, Mat]:
    pos = state[i_pos]
    orientation = state[i_quat]
    orientation_quat = Quaternion(array=orientation)
    tip_pos = pos - orientation_quat.rotate(
        np.array([0, STYLUS_LENGTH, 0]) + IMU_OFFSET
    )
    return (tip_pos, orientation)


def get_orientation_quat(orientation_mat_opencv: Mat):
    return Quaternion(matrix=orientation_mat_opencv).normalised


def nearest_quaternion(reference: Mat, new: Mat):
    """
    Find the sign for new that makes it as close to reference as possible.
    Changing the sign of a quaternion does not change its rotation, but affects
    the difference from the reference quaternion.
    """
    error1 = np.linalg.norm(reference - new)
    error2 = np.linalg.norm(reference + new)
    return (new, error1) if error1 < error2 else (-new, error2)


class DpointFilter:
    history: Deque[HistoryItem]

    def __init__(self, dt):
        self.history = deque()
        self.fs = initial_state()
        self.dt = dt

    def update_imu(self, accel: np.ndarray, gyro: np.ndarray):
        predicted = ekf_predict(self.fs, self.dt, Q)
        self.fs = fuse_imu(predicted, accel, gyro, imu_noise)
        if len(self.history) > smoothing_length:
            self.history.popleft()
        self.history.append(
            HistoryItem(
                self.fs.state,
                self.fs.statecov,
                predicted.state,
                predicted.statecov,
                accel=accel,
                gyro=gyro,
            )
        )

    def update_camera(self, imu_pos: np.ndarray, orientation_mat: np.ndarray) -> list[np.ndarray]:
        if len(self.history) == 0:
            return []
        # Make sure we don't delay past the end of our history
        camera_delay = min(len(self.history) - 1, 4)  # IMU samples

        # Rollback and store recent IMU measurements
        replay: Deque[HistoryItem] = deque()
        for _ in range(camera_delay):
            replay.appendleft(self.history.pop())

        # Fuse camera in its rightful place
        h = self.history[-1]
        fs = FilterState(h.updated_state, h.updated_statecov)
        or_quat = get_orientation_quat(orientation_mat)
        or_quat_smoothed, or_error = nearest_quaternion(
            fs.state[i_quat], or_quat.elements
        )
        pos_error = np.linalg.norm(imu_pos - fs.state[i_pos])
        if pos_error > 0.2 or or_error > 0.3:
            print(f"Resetting state, error: {pos_error + or_error}")
            self.fs = initial_state(imu_pos, or_quat_smoothed)
            self.history = deque()
            return []
        self.fs = fuse_camera(fs, imu_pos, or_quat_smoothed, camera_noise)
        previous = self.history.pop()  # Replace last item
        self.history.append(
            HistoryItem(
                self.fs.state,
                self.fs.statecov,
                previous.predicted_state,
                previous.predicted_statecov,
                # accel=previous.accel,
                # gyro=previous.gyro,
            )
        )

        # Apply smoothing to the rest of the history
        # TODO: We could also smooth the future measurements
        smoothed_estimates = ekf_smooth(
            List(
                [
                    SmoothingHistoryItem(
                        h.updated_state,
                        h.updated_statecov,
                        h.predicted_state,
                        h.predicted_statecov,
                    )
                    for h in self.history
                ]
            ),
            self.dt,
        )

        # Replay the IMU measurements
        predicted_estimates = []
        for item in replay:
            assert item.accel is not None
            assert item.gyro is not None
            self.update_imu(item.accel, item.gyro)
            predicted_estimates.append(self.fs.state)
        return [
            get_tip_pose(state)[0] for state in smoothed_estimates + predicted_estimates
        ]

    def get_tip_pose(self) -> Tuple[Mat, Mat]:
        return get_tip_pose(self.fs.state)
