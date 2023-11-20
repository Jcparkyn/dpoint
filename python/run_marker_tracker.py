from app.marker_tracker import run_tracker

"""
This is a utility script to run the marker tracker without any of the other parts (BLE, filtering, GUI, etc).
"""

if __name__ == "__main__":
    run_tracker(None)