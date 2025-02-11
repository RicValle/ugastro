import ugradio
import numpy as np

#Capture raw 21-cm line data

def capture_21cm_data(filename, duration=60, sample_rate=2.048e6, center_freq=1.42e9, gain=20):
  
    radio = ugradio.sdr.SDR(sample_rate=sample_rate, center_freq=center_freq, gain=gain)
    
    start_time = ugradio.timing.utc()
    julian_date = ugradio.timing.julian_date()
    lst = ugradio.timing.lst()
    
    print(f"Capturing data for {duration} seconds...")
    data = []
    for _ in range(duration):
        samples = radio.capture_data(nsamples=1024)
        data.append(samples)
    
    data = np.array(data)
    np.savez(filename, data=data, start_time=start_time, julian_date=julian_date, lst=lst)
    print(f"Data saved to {filename}")

capture_21cm_data("21cm_data_week1.npz", duration=60)
