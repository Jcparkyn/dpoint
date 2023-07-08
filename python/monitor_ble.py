import struct
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import asyncio
import multiprocessing as mp

import numpy as np


def unpack_imu_data_packet(data: bytearray):
    """Unpacks an IMUDataPacket struct from the given data buffer."""
    ax, ay, az, gx, gy, gz, t = struct.unpack("<6fI", data)
    accel = np.array([ax, ay, az], dtype=np.float32) * 9.8
    gyro = np.array([gx, gy, gz], dtype=np.float32) * np.pi / 180.0
    return (accel, gyro, t)


async def monitor_ble_async(queue: mp.Queue):
    device = await BleakScanner.find_device_by_name("DPOINT")
    if device is None:
        print("could not find device with name DPOINT")
        return
    characteristic = "19B10013-E8F2-537E-4F6C-D104768A1214"

    def queue_notification_handler(
        characteristic: BleakGATTCharacteristic, data: bytearray
    ):
        accel, gyro, t = unpack_imu_data_packet(data)
        queue.put((accel, gyro, t))

    disconnected_event = asyncio.Event()
    async with BleakClient(
        device, disconnected_callback=lambda _: disconnected_event.set()
    ) as client:
        await client.start_notify(characteristic, queue_notification_handler)
        await disconnected_event.wait()
        print("Disconnected from BLE")


def monitor_ble(queue: mp.Queue):
    asyncio.run(monitor_ble_async(queue))
