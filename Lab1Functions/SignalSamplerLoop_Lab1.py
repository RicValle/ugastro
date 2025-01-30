import ugradio
import numpy as np

coeffs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2047])

sample_rates = np.array([1.5e6])
save_path = "../Lab1Data/Section5_2/"
for i, rate in enumerate(sample_rates):
    print(f"Capturing data at {int(rate / 1e3)} kHz sample rate...")
    
    sdrdata = ugradio.sdr.SDR(sample_rate=rate, fir_coeffs=coeffs)  # Initialize SDR
    captured_time_data = sdrdata.capture_data(nblocks = 1, nsamples = 2048)  # Capture data
    sdrdata.close()  # Close SDR before the next iteration
    
    filename = f"{save_path}lab_1_data_480khz_{int(rate / 1e3)}khz.npz"

    np.savez(filename, time_data = captured_time_data, sample_rate = rate)