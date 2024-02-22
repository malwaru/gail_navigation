import pandas as pd
import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import cv2
import PIL, PIL.Image
import io
data_file="./data/output_data.hdf5"
f = h5py.File(data_file, 'r')
print(f"list of kesy {list(f.keys())}")
print(f"list of items {f['odom_data']}")
