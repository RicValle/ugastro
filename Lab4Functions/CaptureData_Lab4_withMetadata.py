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
NSAMPLES = 2048 	    # Number of samples per FFT block
NBLOCKS = 4300			# Number of FFT blocks to average per observation point
CAL_INTERVAL = 4	    # Repeat every N point with calibration diode on 
SAMPLE_RATE = 2.2e6     # Sample rate of SDRs
USB_FREQ = 1420e6       # Center frequency of SDRs
LSB_FREQ = 1420.81150357e6
GAIN = 0                # Internal gain of SDRs
DATE = "4_26_2"         # month_day_attempt
SAVE_BASE_PATH = "./Lab4Data//" + DATE
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
    mode: Literal["LSB", "USB", "cal_on", "init"]
    pointing: ObservationPoint

@dataclass
class DataResult:
    device_index: int
    spectrum: np.ndarray
    mode: Literal["LSB", "USB", "cal_on", "init"]
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

def precompute_observation_plan(mode="grid", num_points=300):
    # Creates list of observation point objects
    plan = []
    id_counter = 0

    if mode == "grid":
        raw_points = []
        for b in np.arange(15, 52, 2):
            delta_l = 2 / np.cos(np.radians(b))
            for l in np.arange(105, 162, delta_l):
                ra, dec = galactic_to_equatorial(l, b)
                point = ObservationPoint(
                    id=id_counter, gal_l=l, gal_b=b, ra=ra, dec=dec,
                    is_calibration=False, mode="grid"
                )
                raw_points.append(point)
                log_queue.put({"event": "Plan Precomputation", "point": point})
                id_counter += 1

        sorted_counter = 0
        with_cal = []
        for i, p in enumerate(raw_points):
            p.id = sorted_counter
            with_cal.append(p)
            if (i + 1) % CAL_INTERVAL == 0:
                cal_p = ObservationPoint(
                    id=p.id, gal_l=p.gal_l, gal_b=p.gal_b, ra=p.ra, dec=p.dec,
                    is_calibration=True, mode="grid"
                )
                with_cal.append(cal_p)
            sorted_counter += 1
        
        plan = sorted(with_cal, key=lambda p:(p.ra, p.dec))

        return plan

    elif mode == "track":
        l, b = 120, 0
        ra, dec = galactic_to_equatorial(l, b)
        while id_counter < num_points:
            point = ObservationPoint(
                id=id_counter, gal_l=l, gal_b=b, ra=ra, dec=dec,
                is_calibration=(id_counter % CAL_INTERVAL == 0),
                mode="track"
            )
            plan.append(point)
            id_counter += 1
        
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

            is_valid = (14 < alt < 85) and (5 < az < 350)
            if not is_valid:
                log_queue.put({"event": "skip", "id": point.id, "reason": "invalid alt/az"})
                failed_queue.put({"event": "skip", "id": point.id, "reason": "invalid alt/az", "alt": alt, "az": az})
                continue
            
            telescope.point(alt, az)
            pointing_done.set()
            log_queue.put({"event": "pointed", "id": point.id, "l": point.gal_l, "b": point.gal_b, "ra":point.ra, "dec":point.dec, "alt": alt, "az": az, "time": datetime.utcnow().isoformat()})
            time.sleep(60)
        except Empty:
            continue

def data_thread(sdr_list: List[sdr.SDR], noise_diode, data_queue, save_queue, log_queue, terminate_flag):
    while not terminate_flag.is_set():
        try:
            task = data_queue.get(timeout=2)
        except Empty:
            continue

        try:
            if task.mode == "cal_on":
                try:
                    noise_diode.on()
                except Exception as e:
                    log_queue.put({"event": "cal_on_error", "message": str(e), "point_id": task.pointing.id})
            else:
                try:
                    noise_diode.off()
                except Exception as e:
                    log_queue.put({"event": "cal_off_error", "message": str(e), "point_id": task.pointing.id})


            for sdr in sdr_list:
                try:
                    if task.mode == "LSB":
                        sdr.set_center_freq(LSB_FREQ)
                    else:
                        sdr.set_center_freq(USB_FREQ)

                    raw = sdr.capture_data(nsamples=NSAMPLES, nblocks=NBLOCKS)
                    avg = average_power_spectrum(raw, direct=sdr.direct)
                    result = DataResult(
                        device_index=sdr.device_index,
                        spectrum=avg,
                        mode=task.mode,
                        pointing=task.pointing,
                        timestamp=datetime.utcnow().isoformat()
                    )
                    if task.mode not in ("cal_off", "init"):
                        save_queue.put(result)
                        log_queue.put({
                            "event": "data_collected",
                            "mode": result.mode,
                            "pointing_id": result.pointing.id,
                            "l": result.pointing.gal_l,
                            "b": result.pointing.gal_b,
                            "device_index": sdr.device_index,
                            "is_calibration": result.pointing.is_calibration,
                            "timestamp": result.timestamp
                        })
                except Exception as e:
                    log_queue.put({"event": "error collecting data", "message": str(e), "id": task.pointing.id})
        except Exception as e:
            log_queue.put({"event": "error interacting with telescope", "message": str(e), "id": task.pointing.id})

def save_thread(save_queue, log_queue, terminate_flag):
    while not terminate_flag.is_set():
        try:
            result = save_queue.get(timeout=2)
            pol_label = POLARIZATION_LABELS.get(result.device_index, f"dev{result.device_index}")
            folder = os.path.join(SAVE_BASE_PATH, pol_label)
            os.makedirs(folder, exist_ok=True)

            fname = os.path.join(folder, f"obs_{result.pointing.id}_{result.mode}.npz")
            np.savez_compressed(
                fname,
                spectrum=result.spectrum,
                gal_l=result.pointing.gal_l,
                gal_b=result.pointing.gal_b,
                ra=result.pointing.ra,
                dec=result.pointing.dec,
                mode=result.mode,
                is_calibration=result.pointing.is_calibration,
                timestamp=result.timestamp
            )
            log_queue.put({
                "event": "saved",
                "file": fname,
                "mode": result.mode,
                "device_index": result.device_index,
                "pointing_id": result.pointing.id,
                "gal_l": result.pointing.gal_l,
                "gal_b": result.pointing.gal_b,
                "is_calibration": result.pointing.is_calibration,
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
# Example run commands in terminal: 
# - "python3 CaptureData_Lab4_roughDraft.py --mode grid"
# - "python3 CaptureData_Lab4_roughDraft.py --mode track --num_points 300"
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDR HI Mapping Script")
    parser.add_argument("--mode", choices=["grid", "track"], default="grid", help="Observation mode")
    parser.add_argument("--num_points", type=int, default=300, help="Number of observation points (not including calibration points)")
    args = parser.parse_args()

    telescope = LeuschTelescope()
    noise_diode = LeuschNoise()
    sdr_list = [
        sdr.SDR(device_index=0, direct=False, center_freq=USB_FREQ, sample_rate=SAMPLE_RATE, gain=GAIN), 
        sdr.SDR(device_index=1, direct=False, center_freq=USB_FREQ, sample_rate=SAMPLE_RATE, gain=GAIN)
    ]

    pointing_queue = Queue()
    data_queue = Queue()
    save_queue = Queue()
    log_queue = Queue()
    failed_queue = Queue()
    terminate_flag = threading.Event()
    pointing_done = threading.Event()

    plan = precompute_observation_plan(mode=args.mode, num_points=args.num_points)

    threading.Thread(target=pointing_thread, args=(telescope, pointing_queue, pointing_done, log_queue, terminate_flag), daemon=True).start()
    threading.Thread(target=data_thread, args=(sdr_list, noise_diode, data_queue, save_queue, log_queue, terminate_flag), daemon=True).start()
    threading.Thread(target=save_thread, args=(save_queue, log_queue, terminate_flag), daemon=True).start()
    threading.Thread(target=log_thread, args=(log_queue, terminate_flag), daemon=True).start()
    
	# Ensure noise diode is OFF before starting
    dummy_point = ObservationPoint(
        id=-1, gal_l=plan[0].gal_l, gal_b=plan[0].gal_b, ra=plan[0].ra, dec=plan[0].dec,
        is_calibration=False, mode="init"
    )
    data_queue.put(DataTask("init", dummy_point))

    try:
        for point in plan:
            pointing_done.clear()
            pointing_queue.put(point)
            pointing_done.wait(timeout=60)

            if point.is_calibration:
                data_queue.put(DataTask("cal_on", point))
            else:
                data_queue.put(DataTask("LSB", point))
                data_queue.put(DataTask("USB", point))

            time.sleep(24)
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping observation...")
    finally:
        terminate_flag.set()
        telescope.stow()
        log_queue.put({"event": "shutdown"})
        print("Waiting for log thread to finish...")
        if not failed_queue.empty():
            with open(os.path.join(SAVE_BASE_PATH, "failed_points.jsonl"), "a") as fail_log:
                while not failed_queue.empty():
                    try:
                        entry = failed_queue.get(timeout=2)
                        entry["time"] = datetime.utcnow().isoformat()
                        fail_log.write(json.dumps(entry) + "\n")
                    except Empty:
                        continue   
        print("Done")