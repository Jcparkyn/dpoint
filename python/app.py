# Parts of this file were scaffolded from https://github.com/vispy/vispy/blob/main/examples/scene/realtime_data/ex03b_data_sources_threaded_loop.py
from collections import deque
from PyQt6 import QtWidgets, QtCore

import vispy
from vispy import scene
from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app
from vispy.util import quaternion
from vispy.visuals import transforms

import time
import numpy as np
import serial
import time
import matlab.engine
import queue
import multiprocessing as mp

from marker_tracker import run_tracker

IMAGE_SHAPE = (600, 800)  # (height, width)
CANVAS_SIZE = (1080, 1080)  # (width, height)
NUM_LINE_POINTS = 400
TRAIL_POINTS = 1000

COLORMAP_CHOICES = ["viridis", "reds", "blues"]
LINE_COLOR_CHOICES = ["black", "red", "blue"]
stylus_len = 0.143


def append_line_point(line: np.ndarray, new_point: np.array):
    """Append new points to a line."""
    line = np.roll(line, -1, axis=0)
    line[-1, :] = new_point
    return line


def get_line_color(line: np.ndarray):
    base_col = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    pos_z = line[:, [2]]
    return np.hstack(
        [
            np.tile(base_col, (line.shape[0], 1)),
            1 - np.clip(pos_z * 200, 0, 1),
        ]
    )


class CanvasWrapper:
    def __init__(self, eng):
        self._eng = eng
        self.canvas = SceneCanvas(size=CANVAS_SIZE, keys="interactive")
        self.canvas.measure_fps()
        self.grid = self.canvas.central_widget.add_grid()

        self.view_top = self.grid.add_view(0, 0, bgcolor="grey")
        self.view_top.camera = scene.TurntableCamera(up="z", fov=0)
        self.cube = visuals.Box(
            0.014, stylus_len, 0.014, edge_color="black", parent=self.view_top.scene
        )
        pen_tip = visuals.XYZAxis(parent=self.cube)
        pen_tip.transform = transforms.MatrixTransform(
            vispy.util.transforms.scale([0.02, 0.02, 0.02])
            @ vispy.util.transforms.translate([0, 0, stylus_len * 0.5])
        )
        self.cube.transform = transforms.MatrixTransform(
            vispy.util.transforms.translate([0, 0, 0])
        )

        trail_data = np.zeros((TRAIL_POINTS, 3), dtype=np.float32)
        self.trail_line = visuals.Line(
            pos=trail_data, color="red", width=2, parent=self.view_top.scene
        )

        axis = scene.visuals.XYZAxis(parent=self.view_top.scene)

    def update_data(self, new_data_dict):
        orientation = new_data_dict["orientation"]
        orientation_quat = quaternion.Quaternion(
            orientation[0], orientation[1], orientation[3], orientation[2]
        )
        or_fix = vispy.util.transforms.rotate(90, [1, 0, 0])
        pos = new_data_dict["position"]
        pos_converted = np.array([-pos[0], pos[1], -pos[2]])
        self.cube.transform.matrix = (
            vispy.util.transforms.translate([0, 0, stylus_len * 0.5])
            @ orientation_quat.get_matrix()
            @ or_fix
            @ vispy.util.transforms.translate(pos_converted)
        )
        self.trail_line.set_data(
            append_line_point(self.trail_line.pos, pos_converted),
            color=get_line_color(self.trail_line.pos),
        )
        # self.canvas.update()


class MainWindow(QtWidgets.QMainWindow):
    closing = QtCore.pyqtSignal()

    def __init__(self, canvas_wrapper: CanvasWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)

        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()

        self._canvas_wrapper = canvas_wrapper
        main_layout.addWidget(self._canvas_wrapper.canvas.native)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def closeEvent(self, event):
        print("Closing main window!")
        self.closing.emit()
        return super().closeEvent(event)


def readVals(device: serial.Serial):
    while device.in_waiting > 6 * 7:
        device.readline()
    line = device.readline().decode().rstrip()
    vals = np.array(line.split(","), dtype=np.float32)
    if vals.size != 6:
        vals = np.zeros(6, dtype=np.float32)
    accel = vals[0:3] * 9.8
    gyro = vals[3:6] * np.pi / 180.0
    # accel = [-accel[2], -accel[1], accel[3]];
    # gyro = [gyro[2], gyro[1], -gyro[3]];
    return accel, gyro


class ImuDataSource(QtCore.QObject):
    new_data = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal()

    def __init__(
        self,
        eng: matlab.engine.MatlabEngine,
        imuf,
        tracker_queue: mp.Queue,
        num_iterations=50000,
        parent=None,
    ):
        super().__init__(parent)
        self._should_end = False
        self._num_iters = num_iterations
        self._eng = eng
        self._imuf = imuf
        self._tracker_queue = tracker_queue

    def run_data_creation(self):
        print("Run data creation is starting")
        device = serial.Serial("COM3", 250000)
        lastTime = time.time()
        task = None
        imu_queue = deque()  # Add artificial delay to match camera feed
        for _ in range(self._num_iters):
            if self._should_end:
                print("Data source saw that it was told to stop")
                break

            try:
                # print(f"Queue size: {self._tracker_queue.qsize()}")
                while self._tracker_queue.qsize() > 3:
                    self._tracker_queue.get()
                camera_position, camera_orientation = self._tracker_queue.get_nowait()
                self._eng.update_tracker(
                    self._imuf, camera_position, camera_orientation, nargout=0
                )
            except queue.Empty:
                pass

            accel, gyro = readVals(device)
            imu_queue.append((accel, gyro))
            if task is not None:
                (position, orientation) = task.result()
                self.new_data.emit(
                    {
                        "position": list(position[0]),
                        "orientation": list(orientation[0]),
                    }
                )

            if len(imu_queue) > 3:
                accel, gyro = imu_queue.popleft()
                currentTime = time.time()
                dt = currentTime - lastTime
                task = self._eng.step(
                    self._imuf, accel, gyro, dt, nargout=2, background=True
                )
                # task.done
                lastTime = currentTime
        print("Data source finishing")
        self.finished.emit()

    def stop_data(self):
        print("Data source is quitting...")
        self._should_end = True


def run_tracker_with_queue(queue: mp.Queue):
    run_tracker(
        lambda orientation, position: queue.put((orientation, position), block=False)
    )


if __name__ == "__main__":
    app = use_app("pyqt6")
    app.create()

    tracker_queue = mp.Queue()
    eng = matlab.engine.connect_matlab()
    canvas_wrapper = CanvasWrapper(eng)
    win = MainWindow(canvas_wrapper)
    data_thread = QtCore.QThread(parent=win)
    # camera_thread = QtCore.QThread(parent=win)
    imuf = eng.ImuFusion()
    data_source = ImuDataSource(eng, imuf, tracker_queue)
    data_source.moveToThread(data_thread)

    camera_process = mp.Process(target=run_tracker_with_queue, args=(tracker_queue,))
    camera_process.start()

    # update the visualization when there is new data
    data_source.new_data.connect(canvas_wrapper.update_data)
    # start data generation when the thread is started
    data_thread.started.connect(data_source.run_data_creation)
    # camera_thread.started.connect(camera_data_source.run_data_creation)
    # if the data source finishes before the window is closed, kill the thread
    data_source.finished.connect(
        data_thread.quit, QtCore.Qt.ConnectionType.DirectConnection
    )
    # if the window is closed, tell the data source to stop
    win.closing.connect(
        data_source.stop_data, QtCore.Qt.ConnectionType.DirectConnection
    )
    # when the thread has ended, delete the data source from memory
    data_thread.finished.connect(data_source.deleteLater)

    win.show()
    data_thread.start()
    # camera_thread.start()
    app.run()

    print("Waiting for data source to close gracefully...")
    data_thread.wait(5000)
