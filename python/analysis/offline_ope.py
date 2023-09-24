import sys
import time
sys.path.append('.')

import glob
import json
import ntpath
import pickle

import cv2
from app.marker_tracker import (
    CameraReading,
    MarkerTracker,
    load_marker_positions,
    read_camera_parameters,
    relative_transform,
)

# This script pre-computes a list of estimated stylus poses from a set of frames.


def main():
    recording_timestamp = "20230905_162344"
    files = glob.glob(f"recordings/{recording_timestamp}/frames/*.bmp")
    with open(
        f"recordings/{recording_timestamp}/camera_extrinsics.pkl", "rb"
    ) as pickle_file:
        baseRvec, baseTvec = pickle.load(pickle_file)
    camera_data = []
    cameraMatrix, distCoeffs = read_camera_parameters("params/camera_params_c922_f30.yml")
    markerPositions = load_marker_positions()

    tracker = MarkerTracker(cameraMatrix, distCoeffs, markerPositions)
    startTime = time.perf_counter()
    processingTimes = []
    for file in files:
        frame_time = int(ntpath.basename(file).split(".")[0])
        frame = cv2.imread(file)
        processingStartTime = time.perf_counter()
        result = tracker.process_frame(frame)
        if result is not None:
            rvec, tvec = result
            rvecRelative, tvecRelative = relative_transform(
                rvec, tvec, baseRvec, baseTvec
            )
            Rrelative = cv2.Rodrigues(rvecRelative)[0]
            camera_data.append((frame_time, CameraReading(tvecRelative, Rrelative)))
        processingEndTime = time.perf_counter()
        processingTimes.append(processingEndTime - processingStartTime)
    endTime = time.perf_counter()
    totalProcessingTime = sum(processingTimes)
    totalTime = endTime - startTime

    print(f"Time per frame: {totalTime / len(files)}, processing only: {totalProcessingTime / len(files)}")
    # with open(f"recordings/{recording_timestamp}/camera_timing.json", "w") as f:
    #     json.dump(processingTimes, f)

    with open(
        f"recordings/{recording_timestamp}/camera_data.json", "w"
    ) as f:
        json.dump([dict(t=t, data=reading.to_json()) for t, reading in camera_data], f, indent=2)


if __name__ == "__main__":
    main()
