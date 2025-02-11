import numpy as np
import ugradio
import matplotlib.pyplot as plt

# Time Conversions

print("Local Time:", ugradio.timing.local_time())
print("UTC Time:", ugradio.timing.utc())
print("Unix Time:", ugradio.timing.unix_time())
print("Julian Date:", ugradio.timing.julian_date())
print("Local Sidereal Time (LST):", ugradio.timing.lst())

#Check SDR signal levels to ensure they are not clipping or too weak

def check_signal_levels():
    
    radio = ugradio.sdr.SDR(sample_rate=2.048e6, center_freq=1.42e9, gain=20)  # 21-cm line frequency, Gain is arbritrary value we need to check and see which value actually works
    samples = radio.capture_data(nsamples=1024)
    hist, bins = np.histogram(samples, bins=50)
    
    plt.figure()
    plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]))
    plt.xlabel("Signal Level")
    plt.ylabel("Count")
    plt.title("Histogram of Signal Levels")
    plt.show()

check_signal_levels()
