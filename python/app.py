# Parts of this file were scaffolded from https://github.com/vispy/vispy/blob/main/examples/scene/realtime_data/ex03b_data_sources_threaded_loop.py
import time
from PyQt6 import QtWidgets, QtCore

import vispy
from vispy import scene
from vispy.io import read_mesh
from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app
from vispy.util import quaternion
from vispy.visuals import transforms

import numpy as np
import queue
import multiprocessing as mp
from filter import DpointFilter

from marker_tracker import run_tracker
from monitor_ble import monitor_ble

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
            1 - np.clip(pos_z * 400, 0, 1),
        ]
    )


class CanvasWrapper:
    def __init__(self):
        self.canvas = SceneCanvas(size=CANVAS_SIZE, keys="interactive")
        self.canvas.measure_fps()
        self.grid = self.canvas.central_widget.add_grid()

        self.view_top = self.grid.add_view(0, 0, bgcolor="grey")
        self.view_top.camera = scene.TurntableCamera(
            up="z",
            fov=0,
            center=(0.10, 0.13, 0),
            elevation=90,
            azimuth=0,
            scale_factor=0.3,
        )
        vertices, faces, normals, texcoords = read_mesh("pen.obj")
        self.pen_mesh = visuals.Mesh(
            vertices, faces, color=(1, 0.5, 0.5, 1), parent=self.view_top.scene
        )
        self.pen_mesh.transform = transforms.MatrixTransform()

        pen_tip = visuals.XYZAxis(parent=self.pen_mesh)
        pen_tip.transform = transforms.MatrixTransform(
            vispy.util.transforms.scale([0.02, 0.02, 0.02])
        )

        trail_data = np.zeros((TRAIL_POINTS, 3), dtype=np.float32)
        self.trail_line = visuals.Line(
            pos=trail_data, color="red", width=2, parent=self.view_top.scene
        )

        axis = scene.visuals.XYZAxis(parent=self.view_top.scene)
        # This is broken for now, see https://github.com/vispy/vispy/issues/2363
        # grid = scene.visuals.GridLines(parent=self.view_top.scene)

    def update_data(self, new_data_dict: dict):
        if "orientation" in new_data_dict:
            orientation = new_data_dict["orientation"]
            orientation_quat = quaternion.Quaternion(*orientation).inverse()
            pos = new_data_dict["position"]
            self.pen_mesh.transform.matrix = (
                orientation_quat.get_matrix() @ vispy.util.transforms.translate(pos)
            )
            self.trail_line.set_data(
                append_line_point(self.trail_line.pos, pos),
                color=get_line_color(self.trail_line.pos),
            )
        elif "position_replace" in new_data_dict:
            position_replace: list[np.ndarray] = new_data_dict["position_replace"]
            if len(position_replace) == 0:
                return
            self.trail_line.pos[-len(position_replace) :, :] = position_replace
            # self.trail_line.set_data(self.trail_line.pos)


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


class SensorDataSource(QtCore.QObject):
    new_data = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal()

    def __init__(
        self,
        tracker_queue: mp.Queue,
        imu_queue: mp.Queue,
        parent=None,
    ):
        super().__init__(parent)
        self._should_end = False
        self._tracker_queue = tracker_queue
        self._imu_queue = imu_queue
        self._filter = DpointFilter(dt=1 / 120)

    def run_data_creation(self):
        print("Run data creation is starting")

        while True:
            if self._should_end:
                print("Data source saw that it was told to stop")
                break

            try:
                while self._tracker_queue.qsize() > 2:
                    self._tracker_queue.get()
                camera_position, camera_orientation = self._tracker_queue.get_nowait()
                tstart = time.time()
                smoothed_tip_pos = self._filter.update_camera(
                    camera_position, camera_orientation
                )
                # print(f"Filter took {(time.time() - tstart)*1000} ms")
                self.new_data.emit({"position_replace": smoothed_tip_pos})
            except queue.Empty:
                pass

            while self._imu_queue.qsize() > 0:
                accel, gyro, t = self._imu_queue.get()
                self._filter.update_imu(accel, gyro)
                position, orientation = self._filter.get_tip_pose()
                self.new_data.emit(
                    {
                        "position": list(position),
                        "orientation": list(orientation),
                    }
                )

        print("Data source finishing")
        self.finished.emit()

    def stop_data(self):
        print("Data source is quitting...")
        self._should_end = True


def run_tracker_with_queue(queue: mp.Queue):
    run_tracker(
        lambda orientation, position: queue.put((position, orientation), block=False)
    )


if __name__ == "__main__":
    np.set_printoptions(
        precision=3, suppress=True, formatter={"float": "{: >5.2f}".format}
    )
    app = use_app("pyqt6")
    app.create()

    tracker_queue = mp.Queue()
    imu_queue = mp.Queue()
    canvas_wrapper = CanvasWrapper()
    win = MainWindow(canvas_wrapper)
    win.resize(*CANVAS_SIZE)
    data_thread = QtCore.QThread(parent=win)
    # camera_thread = QtCore.QThread(parent=win)
    data_source = SensorDataSource(tracker_queue, imu_queue)
    data_source.moveToThread(data_thread)

    camera_process = mp.Process(
        target=run_tracker_with_queue, args=(tracker_queue,), daemon=False
    )
    camera_process.start()
    # imu_process = threading.Thread(target=monitor_ble, args=(imu_queue,), daemon=False)
    imu_process = mp.Process(target=monitor_ble, args=(imu_queue,), daemon=False)
    imu_process.start()

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

    app.run()
    camera_process.terminate()
    imu_process.terminate()
    print("Waiting for data source to close gracefully...")
    data_thread.wait(500)
