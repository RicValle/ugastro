import ugradio
import numpy as np

coeffs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2047])

offset = 0 #Volts
sample_rates = np.array([1e6,1.25e6,1.5e6,1.75e6,2e6,2.25e6,2.5e6,2.75e6,3e6])
save_path = "../Lab1Data/Section5_2/"
for i, rate in enumerate(sample_rates):
    print(f"Capturing data at {int(rate / 1e3)} kHz sample rate...")
    
    sdrdata = ugradio.sdr.SDR(sample_rate=rate, fir_coeffs=coeffs)  # Initialize SDR
    time_data = sdrdata.capture_data(nblocks = 5, nsamples = 2048)  # Capture data
    sdrdata.close()  # Close SDR before the next iteration
    
    filename = f"{save_path}lab_1_data_200khz_{int(rate / 1e3)}khz.npz"

    np.savez(filename, time_data = time_data, sample_rate = rate, offset = offset)

