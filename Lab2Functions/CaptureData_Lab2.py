import ugradio
import numpy as np

# Time Conversions

print("Local Time:", ugradio.timing.local_time())
print("UTC Time:", ugradio.timing.utc())
print("Unix Time:", ugradio.timing.unix_time())
print("Julian Date:", ugradio.timing.julian_date())
print("Local Sidereal Time (LST):", ugradio.timing.lst())

# Capture raw 21-cm line data
azi_ang = #
alt_ang = #

def capture_21cm_data(filename, sample_rate=3.1e6, center_freq=1.42e9, gain=20):
  
    radio = ugradio.sdr.SDR(sample_rate=sample_rate, center_freq=center_freq, gain=gain) # 21-cm line frequency, Gain is arbritrary value we need to check and see which value actually works
    
    start_time = ugradio.timing.utc()
    julian_date = ugradio.timing.julian_date()
    lst = ugradio.timing.lst()
    
    save_path = "../Lab2Data/Section6_2/"
    data = []
    samples = radio.capture_data(nsamples=1024)
    data.append(samples)
    
    data = np.array(data)
    filename = f"{save_path}lab2_data.npz"
    np.savez(filename, data=data, sample_rate = sample_rate, julian_date=julian_date, lst=lst)
    print(f"Data saved to {filename}")

