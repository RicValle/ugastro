import threading
import time
import numpy as np
import json
import argparse
import sys
import os
from queue import Queue, Empty
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Literal, List
from ugradio import timing, coord, leo, sdr
from ugradio.leusch import LeuschTelescope, LeuschNoise
from astropy.coordinates import SkyCoord
import astropy.units as u

# ===============================
# Configuration Parameters
# ===============================
NSAMPLES = 2048 	    # Number of samples per FFT block
NBLOCKS = 17200			# Number of FFT blocks to average per observation point
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
def ts_print(*args, **kwargs):
    print(f"[{datetime.utcnow().isoformat()}]", *args, **kwargs)

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
                if id_counter > num_points:
                    continue
                ra, dec = galactic_to_equatorial(l, b)
                point = ObservationPoint(
                    id=id_counter, gal_l=l, gal_b=b, ra=ra, dec=dec,
                    is_calibration=False, mode="grid"
                )
                raw_points.append(point)
                id_counter += 1

        sorted_counter = 0
        with_cal = []
        for i, p in enumerate(raw_points):
            p.id = sorted_counter
            with_cal.append(p)
            ts_print(f"[PlanGeneration] Point id = {p.id}; (l, b) = ({round(l, 3)}, {round(b, 3)}).")
            if (i + 1) % CAL_INTERVAL == 0:
                cal_p = ObservationPoint(
                    id=p.id, gal_l=p.gal_l, gal_b=p.gal_b, ra=p.ra, dec=p.dec,
                    is_calibration=True, mode="grid"
                )
                with_cal.append(cal_p)
                ts_print(f"[PlanGeneration] Calibration point added.")
            sorted_counter += 1
        
        plan = sorted(with_cal, key=lambda p:(p.ra, p.dec))
        ts_print(f"[PlanGeneration] Point id = {p.id}; (l, b) = ({round(l, 3)}, {round(b, 3)}).")

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
    ts_print("[PointingThread] Started.")
    while not terminate_flag.is_set():
        try:
            point = pointing_queue.get(timeout=2)
            if point is None:
                ts_print("[PointingThread] Got termination signal.")
                break

            jd = timing.julian_date()
            alt, az = coord.get_altaz(point.ra, point.dec, jd)
            ts_print(f"[PointingThread] Trying to point to ID {point.id}: alt={alt:.2f}, az={az:.2f}")

            is_valid = (14 < alt < 85) and (5 < az < 350)
            if not is_valid:
                ts_print(f"[PointingThread] Skipping ID {point.id}: invalid alt/az")
                log_queue.put({"event": "skip", "id": point.id, "reason": "invalid alt/az"})
                failed_queue.put({"event": "skip", "id": point.id, "reason": "invalid alt/az", "alt": alt, "az": az})
                continue
            
            telescope.point(alt, az)
            ts_print(f"[PointingThread] Pointed at ID {point.id} successfully.")
            time.sleep(5)
            pointing_done.set()
            log_queue.put({"event": "pointed", "id": point.id, "l": point.gal_l, "b": point.gal_b, "ra":point.ra, "dec":point.dec, "alt": alt, "az": az, "time": datetime.utcnow().isoformat()})
        except Empty:
            continue
        except Exception as e:
            ts_print(f"[PointingThread] Exception: {e}")
            log_queue.put({"event": "pointing_thread_exception", "error": str(e)})

def data_thread(sdr_list: List[sdr.SDR], noise_diode, data_queue, fft_queue, log_queue, terminate_flag):
    ts_print("[DataThread] Started.")
    while not terminate_flag.is_set():
        try:
            task = data_queue.get(timeout=2)
            if task is None:
                ts_print("[DataThread] Got termination signal.")
                break
        except Empty:
            continue

        try:
            ts_print(f"[DataThread] Processing Task: ID {task.pointing.id}, Mode {task.mode}")
            if task.mode == "cal_on":
                try:
                    ts_print("[DataThread] Turning on noise diode.")
                    noise_diode.on()
                except Exception as e:
                    log_queue.put({"event": "cal_on_error", "message": str(e), "point_id": task.pointing.id})
            else:
                try:
                    ts_print("[DataThread] Turning off noise diode.")
                    noise_diode.off()
                except Exception as e:
                    log_queue.put({"event": "cal_off_error", "message": str(e), "point_id": task.pointing.id})
            ts_print("[DataThread] Diode set.")


            try:
                if task.mode == "LSB":
                    ts_print(f"[DataThread] Configuring SDRs to LSB Freq")
                    sdr_list[0].set_center_freq(LSB_FREQ)
                    sdr_list[1].set_center_freq(LSB_FREQ)
                else:
                    ts_print(f"[DataThread] Configuring SDRs to USB Freq")
                    sdr_list[0].set_center_freq(USB_FREQ)
                    sdr_list[1].set_center_freq(USB_FREQ)

                ts_print("[DataThread] Capturing data...")
                raw_data_dict = sdr.capture_data(sdrs=sdr_list, nsamples=NSAMPLES, nblocks=NBLOCKS)
                ts_print("[DataThread] Data captured. Pushing to FFT queue...")
                
                for device_index, raw_data in raw_data_dict.items():
                    fft_queue.put((device_index, raw_data, task.mode, task.pointing))
                    ts_print(f"[DataThread] Queued device {device_index} for FFT.")
            except Exception as e:
                ts_print(f"[DataThread] Exception during capture: {e}")
                log_queue.put({"event": "error collecting data", "message": str(e), "id": task.pointing.id})
        except Exception as e:
            ts_print(f"[DataThread] Unexpected exception: {e}")
            log_queue.put({"event": "error interacting with telescope", "message": str(e), "id": task.pointing.id})

def fft_thread(fft_queue, save_queue, log_queue, terminate_flag):
    ts_print("[FFTThread] Started.")
    while not terminate_flag.is_set():
        try:
            item = fft_queue.get(timeout=2)
            if item is None:
                ts_print("[FFTThread] Got termination signal.")
                break

            device_index, raw_data, mode, pointing = item
            ts_print(f"[FFTThread] FFT processing for ID {pointing.id}, device {device_index}, mode {mode}")
            avg_spectrum = average_power_spectrum(raw_data)
            ts_print(f"[FFTThread] FFT done.")

            result = DataResult(
                device_index=device_index,
                spectrum=avg_spectrum,
                mode=mode,
                pointing=pointing,
                timestamp=datetime.utcnow().isoformat()
            )

            save_queue.put(result)
            ts_print(f"[FFTThread] Result stored pushed to Save queue.")

            log_queue.put({
                "event": "fft_processed",
                "device_index": device_index,
                "pointing_id": pointing.id,
                "mode": mode,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Empty:
            continue
        except Exception as e:
            ts_print(f"[FFTThread] Exception during FFT: {e}")
            log_queue.put({"event": "fft_thread_exception", "error": str(e)})

def save_thread(save_queue, log_queue, terminate_flag):
    ts_print("[SaveThread] Started.")
    while not terminate_flag.is_set():
        try:
            result = save_queue.get(timeout=2)
            if result is None:
                ts_print("[SaveThread] Got termination signal.")
                break
            
            ts_print("[SaveThread] Received result.")
            pol_label = POLARIZATION_LABELS.get(result.device_index, f"dev{result.device_index}")
            folder = os.path.join(SAVE_BASE_PATH, pol_label)
            os.makedirs(folder, exist_ok=True)

            fname = os.path.join(folder, f"obs_{result.pointing.id}_{result.mode}.npz")
            ts_print(f"[SaveThread] Directory made, attempting save.")
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
            ts_print(f"[SaveThread] Saved {fname}")
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
        except Exception as e:
            ts_print(f"[SaveThread] Exception during save: {e}")
            log_queue.put({"event": "save_thread_exception", "error": str(e)})

def log_thread(log_queue, terminate_flag):
    ts_print("[LogThread] Started.")
    os.makedirs(SAVE_BASE_PATH, exist_ok=True)
    log_path = os.path.join(SAVE_BASE_PATH, "log.jsonl")
    try:
        with open(log_path, "a") as log_file:
            while not terminate_flag.is_set():
                try:
                    entry = log_queue.get(timeout=2)
                    if entry is None:
                        ts_print("[LogThread] Got termination signal.")
                        break

                    entry["time"] = datetime.utcnow().isoformat()
                    log_file.write(json.dumps(entry) + "\n")
                    log_file.flush()  # force writing immediately
                    ts_print(f"[LogThread] Logged event: {entry['event']}")
                except Empty:
                    continue
    except Exception as e:
        ts_print(f"[LogThread] Exception opening/writing log file: {e}")

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
    ts_print(f"[Main] Started script.")

    ts_print(f"[Main] Attempting to connect to Leuschner Telescope & Noise Diode.")
    telescope = LeuschTelescope()
    ts_print(f"[Main] LeuschTelescope connection successful.")
    noise_diode = LeuschNoise()
    ts_print(f"[Main] Noise diode connection successful.")

    ts_print(f"[Main] Attempting SDR initialization.")
    sdr_list = [
        sdr.SDR(device_index=0, direct=False, center_freq=USB_FREQ, sample_rate=SAMPLE_RATE, gain=GAIN), 
        sdr.SDR(device_index=1, direct=False, center_freq=USB_FREQ, sample_rate=SAMPLE_RATE, gain=GAIN)
    ]
    ts_print(f"[Main] SDR initialization successful.")

    pointing_queue = Queue()
    data_queue = Queue()
    fft_queue = Queue()
    save_queue = Queue()
    log_queue = Queue()
    failed_queue = Queue()
    terminate_flag = threading.Event()
    pointing_done = threading.Event()

    ts_print(f"[Main] Generating observation plan.")
    plan = precompute_observation_plan(mode=args.mode, num_points=args.num_points)
    ts_print(f"[Main] Observation plan successfully created with {len(plan)} points.")

    ts_print(f"[Main] Setting up threads.")
    threading.Thread(target=pointing_thread, args=(telescope, pointing_queue, pointing_done, log_queue, terminate_flag), name="PointingThread", daemon=True).start()
    threading.Thread(target=data_thread, args=(sdr_list, noise_diode, data_queue, fft_queue, log_queue, terminate_flag), name="DataThread", daemon=True).start()
    threading.Thread(target=fft_thread, args=(fft_queue, save_queue, log_queue, terminate_flag), name="FFTThread", daemon=True).start()
    threading.Thread(target=save_thread, args=(save_queue, log_queue, terminate_flag), name="SaveThread", daemon=True).start()
    threading.Thread(target=log_thread, args=(log_queue, terminate_flag), name="LogThread", daemon=True).start()
    ts_print(f"[Main] Threads set up successfully.")

	# Ensure noise diode is OFF before starting
    dummy_point = ObservationPoint(
        id=-1, gal_l=plan[0].gal_l, gal_b=plan[0].gal_b, ra=plan[0].ra, dec=plan[0].dec,
        is_calibration=False, mode="init"
    )
    ts_print(f"[Main] Dummy initial point created.")
    data_queue.put(DataTask("init", dummy_point))

    log_queue.put({"event": "Initialization", 
                   "NSAMPLES": NSAMPLES, 
                   "CAL_INTERVAL": CAL_INTERVAL, 
                   "SAMPLE_RATE": SAMPLE_RATE, 
                   "USB_FREQ": USB_FREQ,
                   "LSB_FREQ": LSB_FREQ})

    try:
        for point in plan:
            ts_print(f"[Main] Observation ({round(point.gal_l, 3)}, {round(point.gal_b, 3)}) started.")
            ts_print(f"[Main] Point.is_calibration = {point.is_calibration}")
            pointing_done.clear()
            ts_print(f"[Main] Pointing started.")
            pointing_queue.put(point)
            pointing_done.wait(timeout=60)
            ts_print(f"[Main] Pointing done.")

            if point.is_calibration:
                data_queue.put(DataTask("cal_on", point))
                ts_print(f"[Main] Calibration task added.")
            else:
                data_queue.put(DataTask("LSB", point))
                data_queue.put(DataTask("USB", point))
                ts_print(f"[Main] USB and LSB task added.")

            time.sleep(3)
    except KeyboardInterrupt:
        ts_print("\nInterrupted. Stopping observation...")
    finally:
        terminate_flag.set()
        pointing_queue.put(None)
        data_queue.put(None)
        fft_queue.put(None)
        save_queue.put(None)
        log_queue.put(None)
        try:
            telescope.stow()
        except Exception as e:
            ts_print(f"Error stowing telescope: {e}")
        log_queue.put({"event": "shutdown"})
        ts_print("Waiting for log thread to finish...")
        time.sleep(3)

        for thread in threading.enumerate():
            if thread is not threading.current_thread():
                ts_print(f"Joining thread {thread.name}...")
                thread.join(timeout=10)  # wait up to 10 seconds per thread

        ts_print(f"[Main] Writing skipped points to log file.")
        if not failed_queue.empty():
            with open(os.path.join(SAVE_BASE_PATH, "failed_points.jsonl"), "a") as fail_log:
                while not failed_queue.empty():
                    try:
                        entry = failed_queue.get(timeout=2)
                        entry["time"] = datetime.utcnow().isoformat()
                        fail_log.write(json.dumps(entry) + "\n")
                    except Empty:
                        continue   
        ts_print("Done")