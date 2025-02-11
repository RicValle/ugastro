import numpy as np
import ugradio.timing as timing
import ugradio.sdr as sdr
import time
import datetime
import matplotlib.pyplot as plt

# 1. Time Conversions

print("Local Time:", timing.local_time())
print("UTC Time:", timing.utc())
print("Unix Time:", timing.unix_time())
print("Julian Date:", timing.julian_date())
print("Local Sidereal Time (LST):", timing.lst())

# 2. System Checks

def check_signal_levels():
    #Check SDR signal levels to ensure they are not clipping or too weak
    radio = sdr.SDR(sample_rate=2.048e6, center_freq=1.42e9, gain=20)  # 21-cm line frequency
    samples = radio.capture_data(nsamples=1024)
    hist, bins = np.histogram(samples, bins=50)
    
    plt.figure()
    plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]))
    plt.xlabel("Signal Level")
    plt.ylabel("Count")
    plt.title("Histogram of Signal Levels")
    plt.show()

check_signal_levels()

# 3. Capture 21-cm Data

def capture_21cm_data(filename, duration=60, sample_rate=2.048e6, center_freq=1.42e9, gain=20):
    #Capture raw 21-cm line data and save it with metadata.
    radio = sdr.SDR(sample_rate=sample_rate, center_freq=center_freq, gain=gain)
    
    start_time = timing.utc()
    julian_date = timing.julian_date()
    lst = timing.lst()
    
    print(f"Capturing data for {duration} seconds...")
    data = []
    for _ in range(duration):
        samples = radio.capture_data(nsamples=1024)
        data.append(samples)
        time.sleep(1)  # Ensure 1-second intervals
    
    data = np.array(data)
    np.savez(filename, data=data, start_time=start_time, julian_date=julian_date, lst=lst)
    print(f"Data saved to {filename}")

capture_21cm_data("21cm_data_week1.npz", duration=60)
