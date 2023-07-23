import struct
from typing import NamedTuple
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import asyncio
import multiprocessing as mp

import numpy as np

class StylusReading(NamedTuple):
    accel: np.ndarray
    gyro: np.ndarray
    t: int
    pressure: float

def unpack_imu_data_packet(data: bytearray):
    """Unpacks an IMUDataPacket struct from the given data buffer."""
    ax, ay, az, gx, gy, gz, t, pressure = struct.unpack("<6fIHxx", data)
    accel = np.array([ax, ay, az], dtype=np.float32) * 9.8
    gyro = np.array([gx, gy, gz], dtype=np.float32) * np.pi / 180.0
    return StylusReading(accel, gyro, t, pressure / 2**16)


async def monitor_ble_async(queue: mp.Queue):
    device = await BleakScanner.find_device_by_name("DPOINT")
    if device is None:
        print("could not find device with name DPOINT")
        return
    characteristic = "19B10013-E8F2-537E-4F6C-D104768A1214"

    def queue_notification_handler(
        characteristic: BleakGATTCharacteristic, data: bytearray
    ):
        reading = unpack_imu_data_packet(data)
        queue.put(reading)

    disconnected_event = asyncio.Event()
    async with BleakClient(
        device, disconnected_callback=lambda _: disconnected_event.set()
    ) as client:
        await client.start_notify(characteristic, queue_notification_handler)
        await disconnected_event.wait()
        print("Disconnected from BLE")


def monitor_ble(queue: mp.Queue):
    asyncio.run(monitor_ble_async(queue))
