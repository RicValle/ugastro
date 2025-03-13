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
FOLDER = ""
OBS_NAME = "test_observation"
OBS_TIME = 3600 # seconds
RA = 180 # degrees
DEC = 45 # degrees
OBS_SUN = True  # Bool to measure sun instead of specific coords

date = local_time()
DATE_TIME = date[4:8]+"_"+date[8:10]+"_"+date[11:13]+"-"+date[14:16]
DATA_FILE = os.path.join(FOLDER, f"{OBS_NAME}_data_{DATE_TIME}.npz")
LOG_FILE = os.path.join(FOLDER, f"{OBS_NAME}_log_{DATE_TIME}.json")
BACKUP_FILE = os.path.join(FOLDER, f"{OBS_NAME}_backup_{DATE_TIME}.npz")

# ======= CONFIG ======= 
def log_message(message):
    """Write log messages to the log file."""
    timestamp = utc()
    log_entry = f"[{timestamp}] {message}\n"
    with open(LOG_FILE, "a") as log:
        log.write(log_entry)

# ======= TELESCOPE POINTING ======= #
def point_telescope():
    """Continuously adjust telescope pointing."""
    try:
        while not terminate_flag.is_set():
            jd = julian_date()
            ra, dec = sunpos(jd)
            alt, az = get_altaz(ra, dec, jd)

            ifm.point(alt=alt, az=az, wait=True)
            print(f"Telescope commanded to point to Alt: {alt}, Az: {az}")
            log_message(f"Telescope commanded to point to Alt: {alt}, Az: {az}")
            
            actual_alt, actual_az = ifm.get_pointing()
            print(f"Telescope actually pointing to Alt: {actual_alt}, Az: {actual_az}")
            log_message(f"Telescope actually pointing to Alt: {actual_alt}, Az: {actual_az}")
            time.sleep(5)
    except Exception as e:
        print(f"Telescope error: {e}")
        log_message(f"Telescope error: {e}")

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
        log_message("Spectrometer finished collecting.")
    except Exception as e:
        print(f"Spectrometer error: {e}")
        log_message(f"Spectrometer error: {e}")

# ======= DATA SAVING ======= #
def save_data_periodically():
    """Periodically save collected data to file."""
    try:
        while not terminate_flag.is_set():
            with data_lock:
                if data_buffer:
                    np.savez(DATA_FILE, data_buffer)
                    log_message("Data saved successfully.")
            time.sleep(10)
    except Exception as e:
        print(f"Error saving data: {e}")
        log_message(f"Error saving data: {e}")

# ======= START THREADS ======= #
telescope_thread = threading.Thread(target=point_telescope)
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
finally:
    ifm.stow()
    np.savez(BACKUP_FILE, data_buffer)
    log_message(f"Backup file saved successfully.")

log_message("Saving final dataset...")
with data_lock:
    if data_buffer:
        np.savez(DATA_FILE, data_buffer)
        log_message("Final data saved successfully.")

log_message("Data collection completed.")