from typing import Tuple
import numpy as np
from dataclasses import dataclass
from numpy import typing as npt
from pyquaternion import Quaternion

Mat = npt.NDArray[np.float64]


@dataclass
class FilterState:
    state: Mat
    statecov: Mat


i_quat = [0, 1, 2, 3]
i_av = [4, 5, 6]
i_pos = [7, 8, 9]
i_vel = [10, 11, 12]
i_acc = [13, 14, 15]
i_accbias = [16, 17, 18]
i_gyrobias = [19, 20, 21]

additive_noise = np.zeros(22)
additive_noise[i_pos] = 0
additive_noise[i_vel] = 1e-4
additive_noise[i_acc] = 10
additive_noise[i_av] = 10
additive_noise[i_quat] = 1e-4
additive_noise[i_accbias] = 5e-4
additive_noise[i_gyrobias] = 0
Q = np.diag(additive_noise)

state_size = 22
gravity_vector = np.array([0, 0, 9.81])
accel_noise = 1e-3
gyro_noise = 1e-6
imu_noise = np.diag([accel_noise] * 3 + [gyro_noise] * 3)
camera_noise_pos = 0.5e-5
camera_noise_or = 0.5e-4
camera_noise = np.diag([camera_noise_pos] * 3 + [camera_noise_or] * 4)


def initial_state():
    state = np.zeros(state_size, dtype=np.float64)
    state[i_quat] = [1, 0, 0, 0]
    statecov = np.zeros((state_size, state_size))
    statecov[i_quat, i_quat] = 0.3
    return FilterState(state, statecov)


def state_transition(state: Mat):
    av = state[i_av]
    quat = state[i_quat]
    q0, q1, q2, q3 = quat
    acc = state[i_acc]
    vel = state[i_vel]

    qdot = np.array(
        [
            np.dot(av, [-q1, -q2, -q3]) / 2,
            np.dot(av, [q0, -q3, q2]) / 2,
            np.dot(av, [q3, q0, -q1]) / 2,
            np.dot(av, [-q2, q1, q0]) / 2,
        ]
    )
    pdot = vel
    vdot = acc
    statedot = np.zeros_like(state)
    statedot[i_quat] = qdot
    statedot[i_av] = 0
    statedot[i_pos] = pdot
    statedot[i_vel] = vdot
    statedot[i_acc] = 0
    statedot[i_accbias] = 0
    statedot[i_gyrobias] = 0
    return statedot


def state_transition_jacobian(state: Mat):
    av = state[i_av]
    avx, avy, avz = av
    quat = state[i_quat]
    q0, q1, q2, q3 = quat

    N = len(state)

    # Orientation
    dorientfuncdx = np.zeros((4, N), dtype=state.dtype)
    dorientfuncdx[0, i_quat] = [0, -avx / 2, -avy / 2, -avz / 2]
    dorientfuncdx[0, i_av] = [-q1 / 2, -q2 / 2, -q3 / 2]
    dorientfuncdx[1, i_quat] = [avx / 2, 0, avz / 2, -avy / 2]
    dorientfuncdx[1, i_av] = [q0 / 2, -q3 / 2, q2 / 2]
    dorientfuncdx[2, i_quat] = [avy / 2, -avz / 2, 0, avx / 2]
    dorientfuncdx[2, i_av] = [q3 / 2, q0 / 2, -q1 / 2]
    dorientfuncdx[3, i_quat] = [avz / 2, avy / 2, -avx / 2, 0]
    dorientfuncdx[3, i_av] = [-q2 / 2, q1 / 2, q0 / 2]

    # Position
    dposfuncdx = np.zeros((3, N), dtype=state.dtype)
    dposfuncdx[0, i_vel] = [1, 0, 0]
    dposfuncdx[1, i_vel] = [0, 1, 0]
    dposfuncdx[2, i_vel] = [0, 0, 1]

    # Velocity
    dvelfuncdx = np.zeros((3, N), dtype=state.dtype)
    dvelfuncdx[0, i_acc] = [1, 0, 0]
    dvelfuncdx[1, i_acc] = [0, 1, 0]
    dvelfuncdx[2, i_acc] = [0, 0, 1]

    dfdx = np.zeros((state_size, state_size), dtype=state.dtype)
    dfdx[i_quat, :] = dorientfuncdx
    dfdx[i_pos, :] = dposfuncdx
    dfdx[i_vel, :] = dvelfuncdx
    return dfdx


def imu_measurement(state: Mat):
    av = state[i_av]
    acc = state[i_acc]
    quat = state[i_quat]
    q0, q1, q2, q3 = quat
    accbias = state[i_accbias]

    m_gyro = av + state[i_gyrobias]
    mj_gyro = np.zeros((3, len(state)))
    mj_gyro[:, i_av] = np.eye(3)
    mj_gyro[:, i_gyrobias] = np.eye(3)

    grav = gravity_vector

    m_accel = np.array(
        [
            accbias[0]
            - (acc[0] - grav[0]) * (2 * q0**2 + 2 * q1**2 - 1)
            - (acc[1] - grav[1]) * (2 * q0 * q3 + 2 * q1 * q2)
            + (acc[2] - grav[2]) * (2 * q0 * q2 - 2 * q1 * q3),
            accbias[1]
            - (acc[1] - grav[1]) * (2 * q0**2 + 2 * q2**2 - 1)
            + (acc[0] - grav[0]) * (2 * q0 * q3 - 2 * q1 * q2)
            - (acc[2] - grav[2]) * (2 * q0 * q1 + 2 * q2 * q3),
            accbias[2]
            - (acc[2] - grav[2]) * (2 * q0**2 + 2 * q3**2 - 1)
            - (acc[0] - grav[0]) * (2 * q0 * q2 + 2 * q1 * q3)
            + (acc[1] - grav[1]) * (2 * q0 * q1 - 2 * q2 * q3),
        ]
    )

    mj_accel = np.zeros((3, len(state)))
    mj_accel[0, i_quat] = [
        2 * q2 * (acc[2] - grav[2])
        - 2 * q3 * (acc[1] - grav[1])
        - 4 * q0 * (acc[0] - grav[0]),
        -4 * q1 * (acc[0] - grav[0])
        - 2 * q2 * (acc[1] - grav[1])
        - 2 * q3 * (acc[2] - grav[2]),
        2 * q0 * (acc[2] - grav[2]) - 2 * q1 * (acc[1] - grav[1]),
        -2 * q0 * (acc[1] - grav[1]) - 2 * q1 * (acc[2] - grav[2]),
    ]
    mj_accel[1, i_quat] = [
        2 * q3 * (acc[0] - grav[0])
        - 4 * q0 * (acc[1] - grav[1])
        - 2 * q1 * (acc[2] - grav[2]),
        -2 * q2 * (acc[0] - grav[0]) - 2 * q0 * (acc[2] - grav[2]),
        -2 * q1 * (acc[0] - grav[0])
        - 4 * q2 * (acc[1] - grav[1])
        - 2 * q3 * (acc[2] - grav[2]),
        2 * q0 * (acc[0] - grav[0]) - 2 * q2 * (acc[2] - grav[2]),
    ]
    mj_accel[2, i_quat] = [
        2 * q1 * (acc[1] - grav[1])
        - 2 * q2 * (acc[0] - grav[0])
        - 4 * q0 * (acc[2] - grav[2]),
        2 * q0 * (acc[1] - grav[1]) - 2 * q3 * (acc[0] - grav[0]),
        -2 * q0 * (acc[0] - grav[0]) - 2 * q3 * (acc[1] - grav[1]),
        -2 * q1 * (acc[0] - grav[0])
        - 2 * q2 * (acc[1] - grav[1])
        - 4 * q3 * (acc[2] - grav[2]),
    ]
    mj_accel[0, i_acc] = [
        1 - 2 * q1**2 - 2 * q0**2,
        -2 * q0 * q3 - 2 * q1 * q2,
        2 * q0 * q2 - 2 * q1 * q3,
    ]
    mj_accel[1, i_acc] = [
        2 * q0 * q3 - 2 * q1 * q2,
        1 - 2 * q2**2 - 2 * q0**2,
        -2 * q0 * q1 - 2 * q2 * q3,
    ]
    mj_accel[2, i_acc] = [
        -2 * q0 * q2 - 2 * q1 * q3,
        2 * q0 * q1 - 2 * q2 * q3,
        1 - 2 * q3**2 - 2 * q0**2,
    ]
    mj_accel[:, i_accbias] = np.eye(3)

    m_combined = np.concatenate([m_accel, m_gyro])
    mj_combined = np.vstack([mj_accel, mj_gyro])
    return (m_combined, mj_combined)


def camera_measurement(state: Mat):
    pos = state[i_pos]
    orientation = state[i_quat]
    m_camera = np.concatenate([pos, orientation])
    mj_camera = np.zeros((7, len(state)))
    mj_camera[0:3, i_pos] = np.eye(3)
    mj_camera[3:7, i_quat] = np.eye(4)
    return (m_camera, mj_camera)


def predict_cov_derivative(P: Mat, dfdx: Mat, Q: Mat):
    pDot = dfdx @ P + P @ (dfdx.T) + Q
    pDot = 0.5 * (pDot + pDot.T)
    return pDot


def euler_integrate(x: Mat, xdot: Mat, dt: float):
    return x + xdot * dt


def repair_quaternion(q: Mat):
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def ekf_correct(x: Mat, P: Mat, h: Mat, H: Mat, z: Mat, R: Mat):
    S = H @ P @ H.T + R  # innovation covariance
    W = P @ H.T @ np.linalg.inv(S)
    x = x + W @ (z - h)
    P = P - W @ H @ P
    return x, P


def ekf_predict(fs: FilterState, dt: float):
    xdot = state_transition(fs.state)
    dfdx = state_transition_jacobian(fs.state)
    P = fs.statecov
    Pdot = predict_cov_derivative(P, dfdx, Q)
    state = euler_integrate(fs.state, xdot, dt)
    state[i_quat] = repair_quaternion(state[i_quat])
    statecov = euler_integrate(P, Pdot, dt)

    if np.max(statecov) > 50 or np.max(abs(state)) > 50:
        print("Resetting state")
        return initial_state()

    return FilterState(state, statecov)


def get_orientation_quat(orientation_mat_opencv: Mat):
    orientation_mat = orientation_mat_opencv[:, [2, 1, 0]] * np.array([[-1, 1, 1]]).T
    quat = Quaternion(matrix=orientation_mat).normalised
    # Make sure the scalar component is positive, so that we don't have discontinuities.
    if quat.scalar < 0:
        quat = -quat
    return quat


def fuse_imu(fs: FilterState, accel: np.ndarray, gyro: np.ndarray):
    h, H = imu_measurement(fs.state)
    accel2 = np.array([-accel[1], -accel[0], accel[2]])
    gyro2 = np.array([gyro[1], gyro[0], -gyro[2]])
    z = np.concatenate([accel2, gyro2])  # actual measurement
    state, statecov = ekf_correct(fs.state, fs.statecov, h, H, z, imu_noise)
    state[i_quat] = repair_quaternion(state[i_quat])
    return FilterState(state, statecov)


def fuse_camera(
    fs: FilterState, tip_pos_opencv: np.ndarray, orientation_mat: np.ndarray
):
    h, H = camera_measurement(fs.state)
    or_quat = get_orientation_quat(orientation_mat)
    tip_pos = tip_pos_opencv.flatten() * [1, -1, -1]
    imu_pos = tip_pos - or_quat.rotate(np.array([0, 0.143, 0]))
    z = np.concatenate([imu_pos, or_quat.normalised.elements])  # actual measurement
    state, statecov = ekf_correct(fs.state, fs.statecov, h, H, z, camera_noise)
    state[i_quat] = repair_quaternion(state[i_quat])
    return FilterState(state, statecov)


def get_tip_pose(fs: FilterState) -> Tuple[Mat, Mat]:
    pos = fs.state[i_pos]
    orientation = fs.state[i_quat]
    orientation_quat = Quaternion(array=orientation)
    tip_pos = pos + orientation_quat.rotate(np.array([0, 0.143, 0]))
    return (tip_pos, orientation)
