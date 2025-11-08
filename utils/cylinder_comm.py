import enum
import json
import logging
import time

import numpy as np
import scipy.interpolate as si
import serial
from loguru import logger


class RegisterAddr(enum.IntEnum):
    ID = 2
    Baudrate = 12
    PosGet = 26
    PosSet = 55
    State = 0x62
    Reserve = 0x00


class CommandId(enum.IntEnum):
    ClearError = 0x04
    ReadState = 0x02
    PosCtrlFeedback = 0x21
    PosCtrl = 0x03
    FollowFeedback = 0x20
    Follow = 0x19


def to_signed_16bit(val):
    return val if val < 0x7FFF else val - 0x10000


class CylinderCOMM:
    def __init__(
        self,
        serial_path: str = "/dev/ttyACM0",
        max_q: float = 0.03,
        id: int = 1,
        baudrate: int = 115200,
        is_force_sensing: bool = False,
        width_mapping_path: str = "data/gripper/width_mapping.json",
    ):
        self.serial_path = serial_path
        self.id = id
        self.baudrate = baudrate
        self.serial_path = serial_path
        self.is_force_sensing = is_force_sensing

        self._min_q, self._max_q = 0.0, max_q

        self.width_mapping_path = width_mapping_path

    def start(self):
        self.ser = serial.Serial()
        self.ser.port = self.serial_path
        self.ser.baudrate = self.baudrate
        self.ser.open()
        self._handle_width_mapping(width_mapping_path=self.width_mapping_path)

    def stop(self):
        self.ser.flush()
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        self.ser.close()
        return

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _handle_width_mapping(self, width_mapping_path: str = None):
        if width_mapping_path is None:
            self.g2c_mapping = si.interp1d(
                [self.min_width, self.max_width], [self.min_q, int(self.max_q * 1e3)], kind="linear"
            )
            self.c2g_mapping = si.interp1d(
                [self.min_q, int(self.max_q * 1e3)], [self.min_width, self.max_width], kind="linear"
            )
        else:
            width_mapping = json.load(open(width_mapping_path))
            self.update_width_mapping(width_mapping)

    def update_width_mapping(self, width_mapping: dict):
        # assert width_mapping is not None, "Width mapping is not provided."
        cyl_width, gripper_width = (
            np.array(list(width_mapping.keys())).astype(float) / 1e3,
            np.array(list(width_mapping.values())),
        )
        gripper_width -= gripper_width[0]  # 0 indicates closed gripper (0mm)
        self.g2c_mapping = si.interp1d(
            gripper_width, cyl_width, kind="linear", bounds_error=False, fill_value=(min(cyl_width), max(cyl_width))
        )
        self.c2g_mapping = si.interp1d(
            cyl_width,
            gripper_width,
            kind="linear",
            bounds_error=False,
            fill_value=(min(gripper_width), max(gripper_width)),
        )
        self._min_width, self._max_width = min(gripper_width), max(gripper_width)
        logging.info(f"Loaded width mapping: {width_mapping}")

    def write_register_with_feedback(
        self, cmd: CommandId, addr: RegisterAddr = None, length: int = 1, val: list = None, skip_cksm: bool = False
    ):
        # Write command
        bytes = [0x55, 0xAA]
        bytes.append(length + 2)
        bytes.append(self.id)
        bytes.append(cmd.value)
        if addr is not None:
            bytes.append(addr.value & 0xFF)
        if val is None:
            val = []
        for i in range(length):
            bytes.append(val[i])
        checksum = 0x00
        for i in range(2, len(bytes)):
            checksum += bytes[i]
        checksum &= 0xFF
        bytes.append(checksum)
        self.ser.write(bytes)

        # Read feedback
        time.sleep(0.005)  # necessary?
        recv = self.ser.read_all()
        # Validation
        if len(recv) == 0:
            logger.warning(f"Failed to get response from the cylinder.")
            return []
        elif not (recv[0] == 0xAA and recv[1] == 0x55 and recv[3] == self.id):
            logger.warning(f"Invalid response with unmatched header or id: {recv}")
            return []

        # Read data
        length = (recv[2] & 0xFF) - 2
        val = []
        checksum = 0x00
        for i in range(0, length - 1):
            val.append(recv[7 + i])
        checksum = sum(recv[2:-1]) & 0xFF
        checksum &= 0xFF

        if checksum != recv[-1]:
            # logger.warning(f"Checksum mismatch: {checksum} != {recv[-1]}")
            return []
        return val

    def state_to_info(
        self, state: bytearray
    ):  # target and pos are always floats in [0.0, max_q], which is not expected
        if len(state) == 0:
            return {"valid": False}
        target_q = np.clip(float(state[0] + (state[1] << 8)) / 2000, 0.0, 1.0) * self.max_q
        q = (
            np.clip(
                float(to_signed_16bit(state[2] + (state[3] << 8))) / 2000,
                0.0,
                1.0,
            )
            * self.max_q
        )
        info = {
            "target": target_q,
            "pos": q,
            "width": self.c2g_mapping(q),
            "temperature": state[4],
            "current": (state[5] + (state[6] << 8)) / 1000,
            "error": state[8],
            "valid": True,
        }
        if self.is_force_sensing:
            info["force"] = float(state[7] + (state[9] << 8)) * 0.0098
        return info

    def set_target_with_feedback(self, q: float, control_cylinder: bool = False):
        if not control_cylinder:
            q = self.g2c_mapping(q)
        q = np.clip(q / self.max_q, 0.0, 1.0)
        val = int(q * 2000)
        val = [val & 0xFF, (val >> 8) & 0xFF]
        state = self.write_register_with_feedback(CommandId.FollowFeedback, RegisterAddr.PosSet, 2, val)
        state = bytearray(state)
        return self.state_to_info(state)

    def initialize_with_feedback(self):
        # Clear error before start
        state = self.write_register_with_feedback(CommandId.ClearError, RegisterAddr.Reserve, 1, [0x1E])
        state = bytearray(state)
        return self.state_to_info(state)

    @property
    def min_width(self):
        return self._min_width

    @property
    def max_width(self):
        return self._max_width

    @property
    def min_q(self):
        return self._min_q

    @property
    def max_q(self):
        return self._max_q


if __name__ == "__main__":
    import math

    from tqdm.rich import tqdm_rich as tqdm

    max_q = 0.03
    dt = 0.01
    dq = 5e-5

    with CylinderCOMM("/dev/ttyACM0", max_q, 1) as cyl:
        initial_info = cyl.initialize_with_feedback()
        initial_pos = initial_info["pos"]
        if not math.isclose(initial_pos, 0.0, abs_tol=1e-3):
            raise ValueError("Initial position should be 0.0")

        tqdm.write(f"Initial info: {initial_info}")

        t_start = time.perf_counter()

        # t: 0->5 q: 0.00->0.02
        while True:
            time_passed = time.perf_counter() - t_start
            pos = 0.02 * time_passed / 5
            if pos >= 0.02:
                break
            info = cyl.set_target_with_feedback(pos, True)
            tqdm.write(f"Current info: {info}")
            time.sleep(0.1)

        time.sleep(0.1)
        # t: 0->5 q: 0.02->0.00
        t_start = time.perf_counter()
        while True:
            time_passed = time.perf_counter() - t_start
            pos = 0.02 - 0.02 * time_passed / 5
            if pos <= 0:
                break
            info = cyl.set_target_with_feedback(pos, True)
            tqdm.write(f"Current info: {info}")
            time.sleep(0.1)
