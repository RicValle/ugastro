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
from ugradio import timing, coord, leo, sdr
from ugradio.leusch import LeuschTelescope, LeuschNoise
from astropy.coordinates import SkyCoord
import astropy.units as u

# ===============================
# Configuration Parameters
# ===============================
NSAMPLES = 2048	        # Number of samples per FFT block
NBLOCKS = 4300		    # Number of FFT blocks to average per observation point
CENTER_FREQ = 1420e6    # Center frequency (Hz) of SDRs
SAMPLE_RATE = 2.2e6     # Sample Rate of SDRs
GAIN = 0                # Gain setting of SDRs
CAL_INTERVAL = 4	    # Repeat every N point with calibration diode on 
POINT_DURATION = 7200   # Seconds for observation plan of point (l b) = (120, 0)
FOLDER_NAME = "4_22_1"  # Month_Day_Attempt
SAVE_BASE_PATH = "./Lab4Data//" + FOLDER_NAME
INIT_TIME = time.time()
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

def precompute_observation_plan(mode="grid"):
    # Creates list of observation point objects
    plan = []
    id_counter = 0

    if mode == "grid":
        for b in np.arange(15, 52, 2.4):
            delta_l = 2.4 / np.cos(np.radians(b))
            for l in np.arange(105, 162, delta_l):
                ra, dec = galactic_to_equatorial(l, b)
                point = ObservationPoint(
                    id=id_counter, gal_l=l, gal_b=b, ra=ra, dec=dec,
                    is_calibration=False, mode="grid"
                )
                plan.append(point)
                if id_counter % CAL_INTERVAL == 0:
                    cal_point = ObservationPoint(
                    id=id_counter, gal_l=l, gal_b=b, ra=ra, dec=dec,
                    is_calibration=True, mode="grid"
                    )
                    plan.append(cal_point)
                id_counter += 1

    elif mode == "track":
        l, b = 120, 0
        ra, dec = galactic_to_equatorial(l, b)
        start_time = time.time()
        while time.time() - start_time < POINT_DURATION:
            point = ObservationPoint(
                id=id_counter, gal_l=l, gal_b=b, ra=ra, dec=dec,
                is_calibration=False, mode="track"
            )
            plan.append(point)
            if id_counter % CAL_INTERVAL == 0:
                cal_point = ObservationPoint(
                id=id_counter, gal_l=l, gal_b=b, ra=ra, dec=dec,
                is_calibration=True, mode="grid"
                )
                plan.append(cal_point)
            id_counter += 1
            #time.sleep(15)
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
                log_queue.put({"event": "skip", "id": point.id, "reason": "invalid alt/az", "az": az, "alt": alt})
                failed_queue.put({"event": "skip", "id": point.id, "reason": "invalid alt/az"})
                continue
            telescope.point(alt, az)
            pointing_done.set()
            log_queue.put({"event": "pointed", "id": point.id, "l": point.gal_l, "b": point.gal_b, "alt": alt, "az": az, "time": datetime.utcnow().isoformat()})
        except Empty:
            continue

def data_thread(sdr_list: List[sdr.SDR], noise_diode, data_queue, save_queue, log_queue, terminate_flag):
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
                if task.mode != "cal_off":
                    save_queue.put(result)
                    log_queue.put({
                        "event": "data_collected",
                        "mode": task.mode,
                        "pointing_id": task.pointing.id,
                        "l": task.pointing.gal_l,
                        "b": task.pointing.gal_b,
                        "device_index": sdr.device_index,
                        "is_calibration": task.pointing.is_calibration,
                        "timestamp": result.timestamp
                    })
            except Exception as e:
                log_queue.put({"event": "data error", "message": str(e), "id": task.pointing.id})


def save_thread(save_queue, log_queue, terminate_flag):
    while not terminate_flag.is_set():
        try:
            result = save_queue.get(timeout=2)
            pol_label = POLARIZATION_LABELS.get(result.device_index, f"dev{result.device_index}")
            folder = os.path.join(SAVE_BASE_PATH, pol_label+f"_{INIT_TIME}")
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
        except Exception as e:
            log_queue.put({"event": "failed to save data", "message": str(e), "id": result.pointing.id})

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
# Example run command in terminal: "python3 CaptureData_Lab4_roughDraft.py --mode track --duration 2000"
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDR HI Mapping Script")
    parser.add_argument("--mode", choices=["grid", "track"], default="grid", help="Observation mode")
    args = parser.parse_args()

    telescope = LeuschTelescope()
    noise_diode = LeuschNoise()
    sdr_list = [
        sdr.SDR(device_index=0, direct=False, center_freq=CENTER_FREQ, sample_rate=SAMPLE_RATE, gain=GAIN), 
        sdr.SDR(device_index=1, direct=False, center_freq=CENTER_FREQ, sample_rate=SAMPLE_RATE, gain=GAIN)
    ]

    pointing_queue = Queue()
    data_queue = Queue()
    save_queue = Queue()
    log_queue = Queue()
    failed_queue = Queue()
    terminate_flag = threading.Event()
    pointing_done = threading.Event()

    plan = precompute_observation_plan(mode=args.mode)

    threading.Thread(target=pointing_thread, args=(telescope, pointing_queue, pointing_done, log_queue, terminate_flag), daemon=True).start()
    threading.Thread(target=data_thread, args=(sdr_list, noise_diode, data_queue, save_queue, log_queue, terminate_flag), daemon=True).start()
    threading.Thread(target=save_thread, args=(save_queue, log_queue, terminate_flag), daemon=True).start()
    threading.Thread(target=log_thread, args=(log_queue, terminate_flag), daemon=True).start()
    
	# Ensure noise diode is OFF before starting
    dummy_point = ObservationPoint(
        id=-1, gal_l=0, gal_b=0, ra=0, dec=0,
        is_calibration=False, mode="init"
    )
    data_queue.put(DataTask("cal_off", dummy_point))

    try:
        for point in plan:
            pointing_done.clear()
            pointing_queue.put(point)
            pointing_done.wait(timeout=30)

            if point.is_calibration:
                data_queue.put(DataTask("cal_on", point))
                data_queue.put(DataTask("cal_off", point))

            data_queue.put(DataTask("science", point))
            #time.sleep(60)
            time.sleep(20)
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping observation...")
    finally:
        terminate_flag.set()
        telescope.stow()
        log_queue.put({"event": "shutdown"})
        with open(os.path.join(SAVE_BASE_PATH, "log.jsonl"), "a") as log_file:
            while not log_queue.empty():
                try:
                    entry = log_queue.get(timeout=2)
                    entry["time"] = datetime.utcnow().isoformat()
                    log_file.write(json.dumps(entry) + "\n")
                except Empty:
                    continue
            while not failed_queue.empty():
                try:
                    entry = failed_queue.get(timeout=2)
                    entry["time"] = datetime.utcnow().isoformat()
                    log_file.write(json.dumps(entry) + "\n")
                except Empty:
                    continue
