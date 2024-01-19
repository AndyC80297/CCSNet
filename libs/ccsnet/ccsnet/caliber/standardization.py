import h5py

import numpy as np
import pandas as pd
from pathlib import Path


# import matplotlib.pyplot as plt
# from bokeh.plotting import figure, show
# from bokeh.io import output_notebook

from resample_functions import *

# To do
# Transform it to function or class
# Add I/O Path


# Andersen
family_file = OUTPUT_FILE / "Andresen_2016"
family_file.mkdir(parents=True, exist_ok=True)

for name, file in Andresen_2016.items():
    
    print(name, "  ", file)
    data = np.loadtxt(file)
    # print(data.shape)
    time_o = data[:,0]
    quad_o = data[:,1:]
    
    time, quad = smooth(time_o, quad_o)
    
    qijo = expand_quad_to_10kpc(quad)

    time_cal, qij_cal = sn_resample_quad(time, qijo, 4096)
    
    with h5py.File(family_file / f"{name}.h5", "w") as h:
        
        h.create_dataset("time", data=time_cal)
        h.create_dataset("quad_moment", data=qij_cal)
        
        
family_file = OUTPUT_FILE / "Andresen_2019"
family_file.mkdir(parents=True, exist_ok=True)

for name, file in Andresen_2019.items():
    
    print(name, "  ", file)
    data = np.loadtxt(file)
    # print(data.shape)
    time_o = data[:,0]
    quad_o = data[:,1:]
    
    time, quad = smooth(time_o, quad_o)
    
    qijo = expand_quad_to_10kpc(quad)
    
    time_cal, qij_cal = sn_resample_quad(time, qijo, 4096)
    
    with h5py.File(family_file / f"{name}.h5", "w") as h:
        
        h.create_dataset("time", data=time_cal)
        h.create_dataset("quad_moment", data=qij_cal)
        
        
# OConnor
family_file = OUTPUT_FILE / "OConnor_2018"
family_file.mkdir(parents=True, exist_ok=True)


for name, file in OConnor_2018.items():
    
    data = np.loadtxt(file)
    real_quad = np.empty([len(data), 6])
    
    time_o = data[:,0]
    real_quad[:, 0] = data[:,1]
    real_quad[:, 1] = data[:,3]
    real_quad[:, 2] = data[:,6]
    real_quad[:, 3] = data[:,2]
    real_quad[:, 4] = data[:,4]
    real_quad[:, 5] = data[:,5]
    
    time, quad = smooth(time_o, real_quad)
    
    qijo = expand_quad_to_10kpc(quad)
    
    time_cal, qij_cal = sn_resample_quad(time, qijo, 4096)
    
    with h5py.File(family_file / f"{name}.h5", "w") as h:
        
        h.create_dataset("time", data=time_cal)
        h.create_dataset("quad_moment", data=qij_cal)
        
        
        
# Pan
kpc2cm  = 3.08567758128e+21 # cm
msun_cgs = 1.9885e+33 # g
c_cgs = 2.99792458e10 # cm/s
G_cgs = 6.67430e-8 # cm3 g-1 s-2
factor = G_cgs / c_cgs**4 / (10*kpc2cm) # To 1kpc

family_file = OUTPUT_FILE / "Pan_2021"
family_file.mkdir(parents=True, exist_ok=True)

for name, file in Pan_2021.items():
    
    data = np.genfromtxt(file)
    real_quad = np.empty([len(data), 6])
    time_o = data[:,0]
    real_quad[:, 0] = data[:,1]
    real_quad[:, 1] = data[:,3]
    real_quad[:, 2] = data[:,6]
    real_quad[:, 3] = data[:,2]
    real_quad[:, 4] = data[:,4]
    real_quad[:, 5] = data[:,5]
    
    time, quad = smooth(time_o, real_quad)
    
    qijo = expand_quad_to_10kpc(quad, scale_factor=factor)
    
    time_cal, qij_cal = sn_resample_quad(time, qijo, 4096)

    with h5py.File(family_file / f"{name}.h5", "w") as h:
        
        h.create_dataset("time", data=time_cal[:-1])
        h.create_dataset("quad_moment", data=qij_cal[:-1])
        
        
        
        
# Powell
family_file = OUTPUT_FILE / "Powell_2020"
family_file.mkdir(parents=True, exist_ok=True)

t0 = [-0.318, -0.189, -0.433]
count = 0
for name, file in Powell_2020.items():
    
    data = np.loadtxt(file)
    time = data[:,1] + t0[count]
    quads = data[:,2:11]
    
    time, quads = smooth(time, quads)
    # time = time + t0[count]
    
    qijo = transpose_to_1kpc_quad(time, quads, 3.24078e-23)
    
    time_cal, qij_cal = sn_resample_quad(time, qijo, 4096)
    
    with h5py.File(family_file / f"{name}.h5", "w") as h:
        
        h.create_dataset("time", data=time_cal)
        h.create_dataset("quad_moment", data=qij_cal)

    count += 1
    

# Radice
kpc2cm  = 3.08567758128e+21 # cm
msun_cgs = 1.9885e+33 # g
c_cgs = 2.99792458e10 # cm/s
G_cgs = 6.67430e-8 # cm3 g-1 s-2
factor =  2 * G_cgs*msun_cgs / c_cgs**4 / (10*kpc2cm) # To 1kpc

family_file = OUTPUT_FILE / "Radice_2019"
family_file.mkdir(parents=True, exist_ok=True)

for name, file in Radice_2019.items():

    pd_data = pd.read_csv(file, sep=" ")
    
    pd_data.columns =[
        '1:time', 
        '2:I_xx', '3:I_xy', '4:I_xz', '5:I_yy', '6:I_yz', '7:I_zz', 
        '8:I_dot_xx', '9:I_dot_xy', '10:I_dot_xz', 
        '11:I_dot_yy', '12:I_dot_yz', '13:I_dot_zz', 
        "NA"]
    
    time = pd_data["1:time"].values
    
    _, qxx = smooth(time, pd_data["8:I_dot_xx"].values)
    _, qxy = smooth(time, pd_data["9:I_dot_xy"].values)
    _, qxz = smooth(time, pd_data["10:I_dot_xz"].values)
    _, qyy = smooth(time, pd_data["11:I_dot_yy"].values)
    _, qyz = smooth(time, pd_data["12:I_dot_yz"].values)
    time, qzz = smooth(time, pd_data["13:I_dot_zz"].values)
    qzy = qyz
    qzx = qxz
    qyx = qxy

    # calculating the second time derivative of QM 
    d_qxx = np.gradient(qxx, time) * factor
    d_qxy = np.gradient(qxy, time) * factor
    d_qxz = np.gradient(qxz, time) * factor
    d_qyy = np.gradient(qyy, time) * factor
    d_qyz = np.gradient(qyz, time) * factor
    d_qzz = np.gradient(qzz, time) * factor
    d_qzy = d_qyz
    d_qzx = d_qxz
    d_qyx = d_qxy
    # print(" Ha!? ", f"{file}")
    # time = time[0:len(d_qxx)]
     
    # Create quadrupole moment array for the before processing
    # The quadrupole is given at the source in cm, so it needs to be brought to the distance of 10kpc
    qijo = np.zeros((len(time),3,3),dtype=float)

    qijo[:,0,0] = d_qxx 
    qijo[:,0,1] = d_qxy 
    qijo[:,1,1] = d_qyy 
    qijo[:,2,0] = d_qxz 
    qijo[:,2,1] = d_qyz 
    qijo[:,2,2] = d_qzz 
    qijo[:,1,2] = qijo[:,2,1]
    qijo[:,0,2] = qijo[:,2,0]
    qijo[:,1,0] = qijo[:,0,1]
    
    time_cal, qij_cal = sn_resample_quad(time, qijo, 4096)
    # print(qij_cal)
    with h5py.File(family_file / f"{name}.h5", "w") as h:
        
        h.create_dataset("time", data=time_cal)
        h.create_dataset("quad_moment", data=qij_cal)

# The orignal script can be found in here
# /home/hongyin.chen/Data/3DCCSN_PREMIERE/Resampler.ipynb