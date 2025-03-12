import threading
import time
import json
import ugradio
import os
from ugradio.interf import Interferometer
from snap_spec.snap import UGRadioSnap
from ugradio.sdr import SDR, capture_data
from ugradio.coord import get_altaz, sunpos, precess
from ugradio.timing import local_time, utc, julian_date, lst
import numpy as np

# ======= INITIALIZE ======= #
ifm = Interferometer()
snap = UGRadioSnap(host='localhost', is_discover=True)
snap.initialize(mode='corr', sample_rate=500) #What is the actual sample rate  
data_lock = threading.Lock()
data_buffer = []
terminate_flag = threading.Event()

# ======= CONFIG ======= #
FOLDER = "//Section3//"
OBS_NAME = "test_observation"
OBS_TIME = 3600 # seconds
RA = 180 # degrees
DEC = 45 # degrees
OBS_SUN = False # Bool to measure sun instead of specific coords

DATA_FILE = os.path.join(FOLDER, f"{OBS_NAME}_data.npz") #Changed to NPZ 
LOG_FILE = os.path.join(FOLDER, f"{OBS_NAME}_log.json")

# ======= CONFIG ======= 
def log_message(message):
    """Write log messages to the log file."""
    timestamp = utc()
    log_entry = f"[{timestamp}] {message}\n"
    with open(LOG_FILE, "a") as log:
        log.write(log_entry)

# ======= TELESCOPE POINTING ======= #
def point_telescope(target_alt, target_az):
    """Continuously adjust telescope pointing."""
    try:
        while not terminate_flag.is_set():
            ifm.point(alt=target_alt, az=target_az, wait=True, verbose=True)
            print(f"Telescope pointed to Alt: {target_alt}, Az: {target_az}")
            time.sleep(5)
    except Exception as e:
        print(f"Telescope error: {e}")

# ======= DATA COLLECTION ======= #
def collect_spectrometer_data(duration):
    """Collect data from SNAP spectrometer."""
    prev_cnt = None
    start_time = time.time()
    try:
        while time.time() - start_time < duration and not terminate_flag.is_set():
            data = snap.read_data(prev_cnt)
            if "acc_cnt" in data:
                prev_cnt = data["acc_cnt"]
                with data_lock:
                    data_buffer.append(data)
                log_message(f"Collected spectrometer data. Accumulator count: {prev_cnt}")
            time.sleep(1)
        print("Spectrometer finished collecting.")
    except Exception as e:
        print(f"Spectrometer error: {e}")

# ======= DATA SAVING ======= #
def save_data_periodically():
    """Periodically save collected data to file."""
    try:
        while not terminate_flag.is_set():
            with data_lock:
                if data_buffer:
                    np.save(DATA_FILE, np.array(data_buffer, dtype=object))
                    log_message("Data saved successfully.")
            time.sleep(10)
    except Exception as e:
        log_message(f"Error saving data: {e}")


# ======= SETUP ======= #
try:
    log_message("Initializing observation setup...")
    jd = julian_date()
    if OBS_SUN:
        ra, dec = sunpos(jd)
        log_message("Observing the Sun")
    else:
        ra, dec = precess(RA, DEC, jd)
        log_message(f"Observing RA: {RA}, DEC: {DEC}")
    alt, az = get_altaz(ra, dec, jd)
    log_message(f"Computed Alt: {alt:.2f}, Az: {az:.2f}")
except Exception as e:
    print(f"Input error: {e}")
    exit()

# ======= START THREADS ======= #
telescope_thread = threading.Thread(target=point_telescope, args=(alt, az))
spectrometer_thread = threading.Thread(target=collect_spectrometer_data, args=(OBS_TIME,))
save_thread = threading.Thread(target=save_data_periodically)

for t in [telescope_thread, spectrometer_thread, save_thread]:
    t.daemon = True
    t.start()

# ======= MAIN EXECUTION ======= #
try:
    start_time = time.time()
    log_message("Observation started.")
    
    while time.time() - start_time < OBS_TIME:
        time.sleep(1)

    log_message("Time duration reached, waiting for last block to complete...")
    terminate_flag.set()
    
except KeyboardInterrupt:
    log_message("Observation terminated by user.")
    terminate_flag.set()
except Exception as e:
    log_message(f"Unexpected error: {e}")
    terminate_flag.set()

log_message("Saving final dataset...")
with data_lock:
    if data_buffer:
        np.save(DATA_FILE, np.array(data_buffer, dtype=object))
        log_message("Final data saved successfully.")

log_message("Data collection completed.")
