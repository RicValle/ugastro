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

def observation_worker(plan, telescope, sdr_list, noise_diode, save_queue, log_queue):
    for point in plan:
        jd = timing.julian_date()
        alt, az = coord.get_altaz(point.ra, point.dec, jd, leo.lat, leo.lon, leo.alt)
        if not (14 < alt < 85 and 5 < az < 350):
            log_queue.put({"event": "skip", "id": point.id, "reason": "invalid alt/az", "alt": alt, "az": az})
            continue

        telescope.point(alt, az, wait=True, verbose=True)
        time.sleep(20)
        modes = ["cal_on"] if point.is_calibration else ["LSB", "USB"]

        if point.is_calibration:
            noise_diode.on()
        else:
            noise_diode.off()

        for mode in modes:
            try:
                freq = LSB_FREQ if mode == "LSB" else USB_FREQ
                for s in sdr_list:
                    s.set_center_freq(freq)
                raw_data_dict = sdr.capture_data(sdrs=sdr_list, nsamples=NSAMPLES, nblocks=NBLOCKS)
                for device_index, raw_data in raw_data_dict.items():
                    avg_spectrum = average_power_spectrum(raw_data)
                    result = DataResult(device_index=device_index, spectrum=avg_spectrum, mode=mode, pointing=point, timestamp=datetime.utcnow().isoformat())
                    save_queue.put(result)
                    log_queue.put({"event": "fft_processed", "device_index": device_index, "pointing_id": point.id, "mode": mode})
            except Exception as e:
                log_queue.put({"event": "data_error", "id": point.id, "mode": mode, "error": str(e)})
        time.sleep(3)

def save_thread(save_queue, log_queue, terminate_flag):
    while not terminate_flag.is_set():
        try:
            result = save_queue.get(timeout=2)
            if result is None:
                break
            pol_label = POLARIZATION_LABELS.get(result.device_index, f"dev{result.device_index}")
            folder = os.path.join(SAVE_BASE_PATH, pol_label)
            os.makedirs(folder, exist_ok=True)
            fname = os.path.join(folder, f"obs_{result.pointing.id}_{result.mode}.npz")
            np.savez_compressed(fname, spectrum=result.spectrum, gal_l=result.pointing.gal_l, gal_b=result.pointing.gal_b, ra=result.pointing.ra, dec=result.pointing.dec, mode=result.mode, is_calibration=result.pointing.is_calibration, timestamp=result.timestamp)
            log_queue.put({"event": "saved", "file": fname, "mode": result.mode, "device_index": result.device_index})
        except Empty:
            continue
        except Exception as e:
            log_queue.put({"event": "save_error", "error": str(e)})

def log_thread(log_queue, terminate_flag):
    os.makedirs(SAVE_BASE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_BASE_PATH, "log.jsonl"), "a") as f:
        while not terminate_flag.is_set():
            try:
                entry = log_queue.get(timeout=2)
                if entry is None:
                    break
                entry["time"] = datetime.utcnow().isoformat()
                f.write(json.dumps(entry) + "\n")
                f.flush()
            except Empty:
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["grid", "track"], default="grid")
    parser.add_argument("--num_points", type=int, default=400)
    args = parser.parse_args()

    telescope = LeuschTelescope()
    noise_diode = LeuschNoise()
    sdr_list = [
        sdr.SDR(device_index=0, direct=False, center_freq=USB_FREQ, sample_rate=SAMPLE_RATE, gain=GAIN),
        sdr.SDR(device_index=1, direct=False, center_freq=USB_FREQ, sample_rate=SAMPLE_RATE, gain=GAIN)
    ]

    plan = precompute_observation_plan(mode=args.mode, num_points=args.num_points)
    save_queue, log_queue = Queue(), Queue()
    terminate_flag = threading.Event()

    threading.Thread(target=save_thread, args=(save_queue, log_queue, terminate_flag), daemon=True).start()
    threading.Thread(target=log_thread, args=(log_queue, terminate_flag), daemon=True).start()

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
