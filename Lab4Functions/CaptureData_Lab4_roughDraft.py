import threading
import time
import numpy as np
import json
import argparse
import sys
import os
from queue import Queue, Empty
from datetime import datetime
from dataclasses import dataclass
from typing import Literal, List
from ugradio import timing, coord, leo
from sdr import SDR
from leusch import LeuschTelescope, LeuschNoise
from astropy.coordinates import SkyCoord
import astropy.units as u

# ===============================
# Configuration Parameters
# ===============================
NSAMPLES = 2048
NBLOCKS = 8
CAL_INTERVAL = 4
SAVE_BASE_PATH = "./data"
POLARIZATION_LABELS = {0: "pol0", 1: "pol1"}  # Map device_index to folder/polarization

# ===============================
# Dataclasses for Observation Plan
# ===============================
@dataclass
class ObservationPoint:
    id: int
    gal_l: float
    gal_b: float
    ra: float
    dec: float
    is_calibration: bool
    mode: Literal["grid", "track"]

@dataclass
class DataTask:
    mode: Literal["science", "cal_on", "cal_off"]
    duration: float
    pointing: ObservationPoint

@dataclass
class DataResult:
    device_index: int
    spectrum: np.ndarray
    mode: str
    pointing: ObservationPoint
    timestamp: str

# ===============================
# Helper Functions
# ===============================
def galactic_to_equatorial(l, b):
    c = SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic')
    return c.icrs.ra.deg, c.icrs.dec.deg

def average_power_spectrum(raw_data_blocks: np.ndarray, direct=True) -> np.ndarray:
    if direct:
        fft_blocks = np.fft.rfft(raw_data_blocks, axis=1)
    else:
        complex_blocks = raw_data_blocks[..., 0] + 1j * raw_data_blocks[..., 1]
        fft_blocks = np.fft.fft(complex_blocks, axis=1)
    power_spectra = np.abs(fft_blocks) ** 2
    return np.mean(power_spectra, axis=0)

def precompute_observation_plan(mode="grid", track_duration=3600):
    plan = []
    id_counter = 0

    if mode == "grid":
        for b in np.arange(15, 52, 2):
            delta_l = 2 / np.cos(np.radians(b))
            for l in np.arange(105, 162, delta_l):
                ra, dec = galactic_to_equatorial(l, b)
                point = ObservationPoint(
                    id=id_counter, gal_l=l, gal_b=b, ra=ra, dec=dec,
                    is_calibration=False, mode="grid"
                )
                plan.append(point)
                id_counter += 1

        with_cal = []
        for i, p in enumerate(plan):
            with_cal.append(p)
            if (i + 1) % CAL_INTERVAL == 0:
                cal_p = ObservationPoint(**{**p.__dict__, "id": id_counter, "is_calibration": True})
                with_cal.append(cal_p)
                id_counter += 1
        return with_cal

    elif mode == "track":
        l, b = 130, 40
        ra, dec = galactic_to_equatorial(l, b)
        start_time = time.time()
        while time.time() - start_time < track_duration:
            point = ObservationPoint(
                id=id_counter, gal_l=l, gal_b=b, ra=ra, dec=dec,
                is_calibration=(id_counter % CAL_INTERVAL == 0),
                mode="track"
            )
            plan.append(point)
            id_counter += 1
            time.sleep(60)
        return plan

# ===============================
# Threads
# ===============================
def pointing_thread(telescope, pointing_queue, pointing_done, log_queue, terminate_flag):
    while not terminate_flag.is_set():
        try:
            point = pointing_queue.get(timeout=2)
            jd = timing.julian_date()
            alt, az = coord.get_altaz(point.ra, point.dec, jd)
            is_valid = 14 < alt < 85
            if not is_valid:
                log_queue.put({"event": "skip", "id": point.id, "reason": "invalid alt/az"})
                continue
            telescope.point(alt, az)
            pointing_done.set()
            log_queue.put({"event": "pointed", "id": point.id, "alt": alt, "az": az, "time": datetime.utcnow().isoformat()})
        except Empty:
            continue

def data_thread(sdr_list: List[SDR], noise_diode, data_queue, save_queue, log_queue, terminate_flag):
    while not terminate_flag.is_set():
        try:
            task = data_queue.get(timeout=2)
        except Empty:
            continue

        if task.mode == "cal_on":
            noise_diode.on()
        elif task.mode == "cal_off":
            noise_diode.off()

        for sdr in sdr_list:
            try:
                raw = sdr.capture_data(nsamples=NSAMPLES, nblocks=NBLOCKS)
                avg = average_power_spectrum(raw, direct=sdr.direct)
                result = DataResult(
                    device_index=sdr.device_index,
                    spectrum=avg,
                    mode=task.mode,
                    pointing=task.pointing,
                    timestamp=datetime.utcnow().isoformat()
                )
                save_queue.put(result)
                log_queue.put({
                    "event": "data_collected",
                    "mode": task.mode,
                    "pointing_id": task.pointing.id,
                    "device_index": sdr.device_index,
                    "is_calibration": task.pointing.is_calibration,
                    "timestamp": result.timestamp
                })
            except Exception as e:
                log_queue.put({"event": "error", "message": str(e), "id": task.pointing.id})


def save_thread(save_queue, log_queue, terminate_flag):
    while not terminate_flag.is_set():
        try:
            result = save_queue.get(timeout=2)
            pol_label = POLARIZATION_LABELS.get(result.device_index, f"dev{result.device_index}")
            folder = os.path.join(SAVE_BASE_PATH, pol_label)
            os.makedirs(folder, exist_ok=True)
            fname = os.path.join(folder, f"obs_{result.pointing.id}_{result.mode}.npy")
            np.save(fname, result.spectrum)
            log_queue.put({
                "event": "saved",
                "file": fname,
                "mode": result.mode,
                "device_index": result.device_index,
                "pointing_id": result.pointing.id,
                "timestamp": result.timestamp
            })
        except Empty:
            continue

def log_thread(log_queue, terminate_flag):
    os.makedirs(SAVE_BASE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_BASE_PATH, "log.jsonl"), "a") as log_file:
        while not terminate_flag.is_set():
            try:
                entry = log_queue.get(timeout=2)
                entry["time"] = datetime.utcnow().isoformat()
                log_file.write(json.dumps(entry) + "\n")
            except Empty:
                continue

# ===============================
# Run Script
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDR HI Mapping Script")
    parser.add_argument("--mode", choices=["grid", "track"], default="grid", help="Observation mode")
    parser.add_argument("--duration", type=int, default=3600, help="Duration in seconds (track mode)")
    args = parser.parse_args()

    telescope = LeuschTelescope()
    noise_diode = LeuschNoise()
    sdr_list = [SDR(device_index=0), SDR(device_index=1)]

    pointing_queue = Queue()
    data_queue = Queue()
    save_queue = Queue()
    log_queue = Queue()
    terminate_flag = threading.Event()
    pointing_done = threading.Event()

    plan = precompute_observation_plan(mode=args.mode, track_duration=args.duration)

    threading.Thread(target=pointing_thread, args=(telescope, pointing_queue, pointing_done, log_queue, terminate_flag), daemon=True).start()
    threading.Thread(target=data_thread, args=(sdr_list, noise_diode, data_queue, save_queue, log_queue, terminate_flag), daemon=True).start()
    threading.Thread(target=save_thread, args=(save_queue, log_queue, terminate_flag), daemon=True).start()
    threading.Thread(target=log_thread, args=(log_queue, terminate_flag), daemon=True).start()

    try:
        for point in plan:
            pointing_done.clear()
            pointing_queue.put(point)
            pointing_done.wait(timeout=30)

            if point.is_calibration:
                data_queue.put(DataTask("cal_on", 2, point))
                data_queue.put(DataTask("cal_off", 2, point))

            data_queue.put(DataTask("science", 60, point))
            time.sleep(65)
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping observation...")
    finally:
        terminate_flag.set()
        telescope.stow()
        log_queue.put({"event": "shutdown"})
