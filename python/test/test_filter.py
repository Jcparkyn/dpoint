from approvaltests.approvals import verify
import numpy as np
from filter import (
    initial_state,
    state_transition,
    state_transition_jacobian,
    imu_measurement,
    camera_measurement,
    i_vel,
    i_quat,
    i_pos,
    i_av,
    i_acc,
    state_size,
    FilterState,
)


def initial_state_for_tests():
    state = np.zeros(state_size, dtype=np.float64)
    state[i_quat] = [1, 0, 0, 0]
    statecov = np.eye(state_size) * 0.01
    return FilterState(state, statecov)


def test_initial_state():
    fs = initial_state()
    assert len(fs.state) == 22
    assert fs.statecov.shape == (22, 22)


def test_state_transition_jacobian():
    fs = initial_state_for_tests()
    fs.state[i_vel] = [1, 2, 3]
    fs.state[i_av] = [4, 5, 6]
    fs.state[i_acc] = [7, 8, 9]
    jacobian = state_transition_jacobian(fs.state)
    verify(jacobian)


def test_state_transition():
    fs = initial_state_for_tests()
    fs.state[i_vel] = [1, 2, 3]
    fs.state[i_av] = [4, 5, 6]
    fs.state[i_acc] = [7, 8, 9]
    st = state_transition(fs.state)
    verify(st)


def test_imu_measurement():
    fs = initial_state_for_tests()
    fs.state[i_vel] = [1, 2, 3]
    fs.state[i_av] = [4, 5, 6]
    fs.state[i_acc] = [7, 8, 9]
    measurement = imu_measurement(fs.state)
    verify(measurement)


def test_camera_measurement():
    fs = initial_state_for_tests()
    fs.state[i_pos] = [1, 2, 3]
    fs.state[i_quat] = [4, 5, 6, 7]
    measurement = camera_measurement(fs.state)
    verify(measurement)
