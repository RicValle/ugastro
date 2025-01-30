# Test file; doesn't work yet
import ugradio
import numpy as np

coeffs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2047])

# First round: 400 kHz signal
# Second round: 1.5 MHz signal

#sdr_filtered_zone_1 = ugradio.sdr.SDR(sample_rate=3.1e6, fir_coeffs=coeffs) # 3.1 MHz
#time_data_zone_1 = sdr_filtered_zone_1.capture_data()
#
#sdr_filtered_zone_2 = ugradio.sdr.SDR(sample_rate=2e6, fir_coeffs=coeffs) # 2 MHz
#time_data_zone_2 = sdr_filtered_zone_2.capture_data()
# 
sdr_filtered_zone_3 = ugradio.sdr.SDR(sample_rate=1.3e6, fir_coeffs=coeffs) # 1.3 MHz
time_data_zone_3 = sdr_filtered_zone_3.capture_data()
# 
np.savez("lab_1_data_1500khz_1300khz", time_data_zone_3) #, time_data_zone_2, time_data_zone_3)
