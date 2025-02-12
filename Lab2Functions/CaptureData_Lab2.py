import ugradio
import numpy as np

# Time Conversions

print("Local Time:", ugradio.timing.local_time())
print("UTC Time:", ugradio.timing.utc())
print("Unix Time:", ugradio.timing.unix_time())
print("Julian Date:", ugradio.timing.julian_date())
print("Local Sidereal Time (LST):", ugradio.timing.lst())

# Capture raw 21-cm line data
azi_ang = #degrees
alt_ang = #degrees
sample_rate = 3.1e6 #Hz
center_freq = 1.420405e9 #Hz
gain = 0 # ?

utc = ugradio.timing.utc()
pst = ugradio.timing.pst()
jd = ugradio.timing.julian_date()
lst = ugradio.timing.lst()

save_path = "../Lab2Data/Section6_2/6_1_1"

sdrdata = ugradio.sdr.SDR(sample_rate=sample_rate, center_freq=center_freq, gain=gain, direct = False) # 21-cm line frequency, Gain is arbritrary value we need to check and see which value actually works

time_data = radio.capture_data(nblocks = 51, nsamples=8192)

sdrdata.close()

filename = f"{save_path}lab_2_.npz"

np.savez(filename, time_data=time_data, sample_rate = sample_rate, center_freq = center_freq, gain = gain , utc = utc, pst = pst, jd = jd, lst = lst)
