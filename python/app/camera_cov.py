import numpy as np
import cv2 as cv
from pyquaternion import Quaternion


def projection_matrix(rvec, tvec, camera_matrix):
    R = cv.Rodrigues(rvec)[0]
    result = np.matmul(camera_matrix, np.hstack((R, tvec)))
    assert result.shape == (3, 4)
    return result


def point_dWorld_dPose(q_object: np.ndarray, object_point: np.ndarray):
    assert q_object.shape == (4,)
    assert len(object_point) == 3
    o1, o2, o3 = object_point.flatten()
    q1, q2, q3, q4 = q_object

    # fmt: off
    return np.array([
        [2*q3*o3 - 2*q4*o2, 2*q3*o2 + 2*q4*o3, 2*q1*o3 + 2*q2*o2 - 4*q3*o1, 2*q2*o3 - 2*q1*o2 - 4*q4*o1, 1, 0, 0],
        [2*q4*o1 - 2*q2*o3, 2*q3*o1 - 4*q2*o2 - 2*q1*o3, 2*q2*o1 + 2*q4*o3, 2*q1*o1 + 2*q3*o3 - 4*q4*o2, 0, 1, 0],
        [2*q2*o2 - 2*q3*o1, 2*q1*o2 - 4*q2*o3 + 2*q4*o1, 2*q4*o2 - 4*q3*o3 - 2*q1*o1, 2*q2*o1 + 2*q3*o2, 0, 0, 1],
        [0] * 7,
    ])
    # fmt: on


def duv_dxyz(xyz):
    x, y, z = xyz.flatten()
    return np.array([[1 / z, 0, -x / z**2], [0, 1 / z, -y / z**2]])


def point_dUV_dPose(
    q_object: np.ndarray,
    t_object: np.ndarray,
    object_point: np.ndarray,
    proj_matrix: np.ndarray,
):
    # print("p", object_point)
    dxyz_dPose = proj_matrix @ point_dWorld_dPose(q_object, object_point)
    R = Quaternion(q_object).rotation_matrix
    world_point = R @ object_point.reshape(3, 1) + t_object.reshape(3, 1)
    assert world_point.shape == (3, 1)
    xyz = proj_matrix @ np.vstack((world_point, 1))
    return duv_dxyz(xyz) @ dxyz_dPose


def df_dPose(
    q_object: np.ndarray,
    t_object: np.ndarray,
    object_points: list[np.ndarray],
    proj_matrix: np.ndarray,
):
    assert q_object.shape == (4,)
    result = np.vstack([
        point_dUV_dPose(q_object, t_object, p, proj_matrix) for p in object_points
    ])
    assert result.shape == (2 * len(object_points), 7)
    return result


def camera_measurement_cov(
    q_object: np.ndarray,
    t_object: np.ndarray,
    object_points: np.ndarray,  # one row per point
    camera_rvec: np.ndarray,
    camera_tvec: np.ndarray,
    camera_matrix: np.ndarray,
    corner_stdev: float,
) -> np.ndarray:
    proj_matrix = projection_matrix(camera_rvec, camera_tvec, camera_matrix)
    J = df_dPose(q_object, t_object, object_points, proj_matrix)
    variance = corner_stdev**2
    return variance * np.linalg.pinv(J.T @ J)
