import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extractData(filename):
    ''' Extracting captured data from the lab1 npz array from SDR'''
    raw_data = np.load(filename)
    return raw_data['arr_0'][0]


