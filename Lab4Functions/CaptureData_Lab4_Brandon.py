import threading
import time
import numpy as np
import json
import argparse
import os
from queue import Queue, Empty
from datetime import datetime
from dataclasses import dataclass
from typing import Literal, List
from ugradio import timing, coord, leo, sdr
from ugradio.leusch import LeuschTelescope, LeuschNoise
from astropy.coordinates import SkyCoord
import astropy.units as u

NSAMPLES = 2048
NBLOCKS = 17200
CAL_INTERVAL = 4
SAMPLE_RATE = 2.2e6
USB_FREQ = 1420e6
LSB_FREQ = 1420.81150357e6
GAIN = 0
DATE = "4_28_1"
SAVE_BASE_PATH = "./Lab4Data//" + DATE
POLARIZATION_LABELS = {0: "pol0", 1: "pol1"}

CUSTOM_POINTS = [
    (120.0, 20.0),  
    (120.0, 20.0) 
    # Add (l, b) pairs as needed
]

@dataclass
class ObservationPoint:
    id: int
    gal_l: float
    gal_b: float
    ra: float
    dec: float
    is_calibration: bool
    mode: Literal["grid", "track", "custom"]

@dataclass
class DataResult:
    device_index: int
    spectrum: np.ndarray
    mode: Literal["LSB", "USB", "cal_on"]
    pointing: ObservationPoint
    timestamp: str

def ts_print(*args, **kwargs):
    print(f"[{datetime.utcnow().isoformat()}]", *args, **kwargs)

def galactic_to_equatorial(l, b):
    c = SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic')
    return c.icrs.ra.deg, c.icrs.dec.deg

def average_power_spectrum(raw_data_blocks: np.ndarray, direct=False) -> np.ndarray:
    if direct:
        fft_blocks = np.fft.rfft(raw_data_blocks, axis=1)
    else:
        complex_blocks = raw_data_blocks[..., 0] + 1j * raw_data_blocks[..., 1]
        fft_blocks = np.fft.fft(complex_blocks, axis=1)
    power_spectra = np.abs(fft_blocks) ** 2
    return np.mean(power_spectra, axis=0)

def precompute_observation_plan(mode="grid", num_points=300):
    plan, id_counter = [], 0
    if mode == "grid":
        raw_points = []
        for b in np.arange(15, 52, 2):
            delta_l = 2 / np.cos(np.radians(b))
            for l in np.arange(105, 162, delta_l):
                if id_counter > num_points:
                    continue
                ra, dec = galactic_to_equatorial(l, b)
                raw_points.append(ObservationPoint(id=id_counter, gal_l=l, gal_b=b, ra=ra, dec=dec, is_calibration=False, mode="grid"))
                id_counter += 1
        sorted_counter, with_cal = 0, []
        for i, p in enumerate(raw_points):
            p.id = sorted_counter
            with_cal.append(p)
            if (i + 1) % CAL_INTERVAL == 0:
                cal_p = ObservationPoint(id=p.id, gal_l=p.gal_l, gal_b=p.gal_b, ra=p.ra, dec=p.dec, is_calibration=True, mode="grid")
                with_cal.append(cal_p)
            sorted_counter += 1
        return sorted(with_cal, key=lambda p: (p.ra, p.dec))
    
    elif mode == "track":
        l, b = 120, 0
        ra, dec = galactic_to_equatorial(l, b)
        while id_counter < num_points:
            plan.append(ObservationPoint(id=id_counter, gal_l=l, gal_b=b, ra=ra, dec=dec, is_calibration=(id_counter % CAL_INTERVAL == 0), mode="track"))
            id_counter += 1
        return plan
    
    elif mode == "custom":
        plan = []
        for i, (l, b) in enumerate(CUSTOM_POINTS):
            ra, dec = galactic_to_equatorial(l, b)
            plan.append(ObservationPoint(id=i, gal_l=l, gal_b=b, ra=ra, dec=dec, is_calibration=(i % CAL_INTERVAL == 0), mode="custom"))
            if (i + 1) % CAL_INTERVAL == 0:
                plan.append(ObservationPoint(id=i, gal_l=l, gal_b=b, ra=ra, dec=dec, is_calibration=True, mode="custom"))
        return plan

def observation_worker(plan, telescope, sdr_list, noise_diode, save_queue, log_queue):
    ts_print("[Worker] Started.")
    for point in plan:
        try:
            ts_print(f"[Worker] Starting observation for Point ID {point.id} (l={point.gal_l:.2f}, b={point.gal_b:.2f})")
            jd = timing.julian_date()
            alt, az = coord.get_altaz(point.ra, point.dec, jd, leo.lat, leo.lon, leo.alt)
            ts_print(f"[Worker] Calculated alt={alt:.2f}, az={az:.2f} for RA={point.ra:.2f}, Dec={point.dec:.2f}")

            if not (14 < alt < 85 and 5 < az < 350):
                ts_print(f"[Worker] Skipping point ID {point.id} due to invalid pointing limits.")
                log_queue.put({"event": "skip", "id": point.id, "reason": "invalid alt/az", "alt": alt, "az": az})
                continue
        except Exception as e:
            ts_print(f"[Worker] ERROR calculating alt/az for Point ID {point.id}: {e}")
            log_queue.put({"event": "pointing_error", "id": point.id, "error": str(e)})
            continue

        try:
            ts_print(f"[Worker] Pointing telescope to alt={alt:.2f}, az={az:.2f} for Point ID {point.id}")
            telescope.point(alt, az, wait=True, verbose=True)
            time.sleep(2)
        except Exception as e:
            ts_print(f"[Worker] ERROR pointing telescope for Point ID {point.id}: {e}")
            log_queue.put({"event": "pointing_error", "id": point.id, "error": str(e)})
            continue

        try:
            modes = ["cal_on"] if point.is_calibration else ["LSB", "USB"]

            ts_print(f"[Worker] Interacting with noise diode for Point ID {point.id} in modes: {modes}")
            if point.is_calibration:
                noise_diode.on()
            else:
                noise_diode.off()
        except Exception as e:
            ts_print(f"[Worker] ERROR interacting with noise diode for Point ID {point.id}: {e}")
            log_queue.put({"event": "noise_diode_error", "id": point.id, "error": str(e)})
            continue
    
        try:
            for mode in modes:
                try:
                    freq = LSB_FREQ if mode == "LSB" else USB_FREQ
                    ts_print(f"[Worker] Starting {mode} capture for Point ID {point.id}")

                    for s in sdr_list:
                        s.set_center_freq(freq)
                        ts_print(f"[Worker] SDR {s.device_index} set to freq={freq/1e6:.3f} MHz")

                    raw_data_dict = sdr.capture_data(sdrs=sdr_list, nsamples=NSAMPLES, nblocks=NBLOCKS)
                    ts_print(f"[Worker] Capture complete for mode={mode} at Point ID {point.id}")

                    for device_index, raw_data in raw_data_dict.items():
                        ts_print(f"[Worker] Computing FFT for SDR {device_index}, mode={mode}, Point ID {point.id}")
                        avg_spectrum = average_power_spectrum(raw_data)
                        result = DataResult(device_index=device_index, spectrum=avg_spectrum, mode=mode, pointing=point, timestamp=datetime.utcnow().isoformat())
                        
                        save_queue.put(result)
                        log_queue.put({"event": "fft_processed", "device_index": device_index, "pointing_id": point.id, "mode": mode})
                        ts_print(f"[Worker] Average FFT saved for SDR {device_index}, mode={mode}, Point ID {point.id}")
                except Exception as e:
                    log_queue.put({"event": "data_error", "id": point.id, "mode": mode, "error": str(e)})
            time.sleep(3)
        except Exception as e:
            ts_print(f"[Worker] ERROR during {mode} capture or FFT at Point ID {point.id}: {e}")
            log_queue.put({"event": "data_error", "id": point.id, "mode": mode, "error": str(e)})

def save_thread(save_queue, log_queue, terminate_flag):
    ts_print("[SaveThread] Started.")
    while not terminate_flag.is_set():
        try:
            result = save_queue.get(timeout=2)
            if result is None:
                ts_print("[SaveThread] Received termination signal.")
                break
            pol_label = POLARIZATION_LABELS.get(result.device_index, f"dev{result.device_index}")
            folder = os.path.join(SAVE_BASE_PATH, pol_label)
            os.makedirs(folder, exist_ok=True)
            fname = os.path.join(folder, f"obs_{result.pointing.id}_{result.mode}.npz")
            ts_print(f"[SaveThread] Saving {fname}")
            np.savez_compressed(fname, spectrum=result.spectrum, gal_l=result.pointing.gal_l, gal_b=result.pointing.gal_b, ra=result.pointing.ra, dec=result.pointing.dec, mode=result.mode, is_calibration=result.pointing.is_calibration, timestamp=result.timestamp)
            log_queue.put({"event": "saved", "file": fname, "mode": result.mode, "device_index": result.device_index})
            ts_print(f"[SaveThread] Saved {fname}")
        except Empty:
            continue
        except Exception as e:
            log_queue.put({"event": "save_error", "error": str(e)})

def log_thread(log_queue, terminate_flag):
    ts_print("[LogThread] Started.")
    os.makedirs(SAVE_BASE_PATH, exist_ok=True)
    try:
        with open(os.path.join(SAVE_BASE_PATH, "log.jsonl"), "a") as f:
            while not terminate_flag.is_set():
                try:
                    entry = log_queue.get(timeout=2)
                    if entry is None:
                        ts_print("[LogThread] Received termination signal.")
                        break
                    entry["time"] = datetime.utcnow().isoformat()
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                except Empty:
                    continue
    except Exception as e:
        ts_print(f"[LogThread] ERROR: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["grid", "track", "custom"], default="grid")
    parser.add_argument("--num_points", type=int, default=400)
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

    ts_print(f"[Main] Generating observation plan.")
    plan = precompute_observation_plan(mode=args.mode, num_points=args.num_points)
    ts_print(f"[Main] Observation plan successfully created with {len(plan)} points.")

    save_queue, log_queue = Queue(), Queue()
    terminate_flag = threading.Event()

    ts_print(f"[Main] Setting up threads.")
    threading.Thread(target=save_thread, args=(save_queue, log_queue, terminate_flag), daemon=True).start()
    threading.Thread(target=log_thread, args=(log_queue, terminate_flag), daemon=True).start()
    ts_print(f"[Main] Threads set up successfully.")

    log_queue.put({"event": "Initialization", 
        "NBLOCKS": NBLOCKS,
        "NSAMPLES": NSAMPLES, 
        "CAL_INTERVAL": CAL_INTERVAL, 
        "SAMPLE_RATE": SAMPLE_RATE, 
        "USB_FREQ": USB_FREQ,
        "LSB_FREQ": LSB_FREQ,
        "GAIN": GAIN,})

    try:
        observation_worker(plan, telescope, sdr_list, noise_diode, save_queue, log_queue)
    except KeyboardInterrupt:
        ts_print("Interrupted.")
    finally:
        terminate_flag.set()
        save_queue.put(None)
        log_queue.put(None)
        try:
            telescope.stow()
            noise_diode.off()
        except Exception as e:
            ts_print(f"[Shutdown] Exception: {e}")
        time.sleep(3)
        ts_print("Shutdown complete.")
