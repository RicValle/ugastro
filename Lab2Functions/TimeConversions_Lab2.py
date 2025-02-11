import numpy as np
import ugradio
import matplotlib.pyplot as plt

# 1. Time Conversions

print("Local Time:", ugradio.timing.local_time())
print("UTC Time:", ugradio.timing.utc())
print("Unix Time:", ugradio.timing.unix_time())
print("Julian Date:", ugradio.timing.julian_date())
print("Local Sidereal Time (LST):", ugradio.timing.lst())
