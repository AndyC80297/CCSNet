import h5py
import numpy as np
from scipy import interpolate

def smooth(
    time,
    data
):
    # Remove duplicate 
    time_1, uni_idx = np.unique(time, return_index=True)
    if len(time_1) != len(time):
        print("Canceled")
    data = data[uni_idx]
    
    # Sort array by time 
    sorted_idx = np.argsort(time_1)
    time_2 = time_1[sorted_idx]
    if not (time_1 == time_2).all():
        print("Sorted")
    data = data[sorted_idx]
    
    return time_2, data


def expand_quad_to_10kpc(
    data,
    scale_factor: float=3.2407792896664e-23
):
    """_summary_

    Args:
        data (_type_): _description_
        scale_factor (float): Resacle the second dirvitave 
        quardrupole moment to 1kPc.
        cm2kpc = 3.2407792896664E-22
        For Pan it's 5.35553002305461e-69

    Returns:
        _type_: _description_
    """
    qijo = np.empty([data.shape[0],3, 3])
    
    qijo[:,0,0] = data[:,0] * scale_factor
    qijo[:,1,1] = data[:,1] * scale_factor
    qijo[:,2,2] = data[:,2] * scale_factor
    qijo[:,0,1] = data[:,3] * scale_factor
    qijo[:,0,2] = data[:,4] * scale_factor
    qijo[:,1,2] = data[:,5] * scale_factor
    qijo[:,2,1] = qijo[:,1,2]
    qijo[:,2,0] = qijo[:,0,2]
    qijo[:,1,0] = qijo[:,0,1]
    
    return qijo

def transpose_to_1kpc_quad(
    time,
    quads,
    scale_factor: float=3.24078e-22
):
    
    # time = d[:,1]
    # quads = d[:,2:11]
    # d=np.loadtxt(filename) 

    nn=len(time)
    a = 0.5 * (
        np.reshape(quads,(nn,3,3)) 
        + np.transpose(np.reshape(quads,(nn,3,3)),(0,2,1))
    )

    qijo = np.zeros((len(time),3,3),dtype=float)

    qijo[:,0,0] = a[:,0,0]  * scale_factor
    qijo[:,0,1] = a[:,0,1]  * scale_factor
    qijo[:,1,1] = a[:,1,1]  * scale_factor
    qijo[:,2,0] = a[:,2,0]  * scale_factor
    qijo[:,2,1] = a[:,2,1]  * scale_factor
    qijo[:,2,2] = a[:,2,2]  * scale_factor
    qijo[:,1,2] = qijo[:,2,1]
    qijo[:,0,2] = qijo[:,2,0]
    qijo[:,1,0] = qijo[:,0,1]
    
    return qijo

def sn_resample_wave(t,h,fs):
    """
    Interpolate array h to the fs sampling frequency.
    
    Input:
        t  - time array, in seconds
        h  - strain array to be interpolated
        fs - sampling frequency
    Output:
        t1 - time array, after resampling
        h1 - new strain array
    """
    
    # Quick check
    if len(t)!=len(h):
        print("Error: t and h need to have equal sizes")
        return 0
    
    # Define new time with fs
    t1 = np.arange(t[0],t[-1],1.0/fs)
    
    # Interpolation
    tck = interpolate.splrep(t,h,s=0)
    h1  = interpolate.splev(t1,tck,der=0)
    
    return t1, h1


def sn_resample_quad(t,qij,fs):
    """
    Interpolate each component of quadrupole moment to the fs sampling frequency.
    
    Input:
        t   - time array, in seconds
        qij - quadrupole moment N x 3 x 3 array to be interpolated
        fs  - sampling frequency
    Output:
        t1   - time array, after resampling
        qij1 - new quadrupole moment array
    """

    # Find new time array
    t1,tmp = sn_resample_wave(t,qij[:,0,0],fs)

    # Define new quadrupole moment array
    qij1 = np.zeros((len(t1),3,3),dtype=float)

    # Interpolate each quadrupole moment
    t1, qij1[:,0,0] = sn_resample_wave(t, qij[:,0,0], fs)
    t1, qij1[:,0,1] = sn_resample_wave(t, qij[:,0,1], fs)
    t1, qij1[:,0,2] = sn_resample_wave(t, qij[:,0,2], fs)
    t1, qij1[:,1,1] = sn_resample_wave(t, qij[:,1,1], fs)
    t1, qij1[:,1,2] = sn_resample_wave(t, qij[:,1,2], fs)
    t1, qij1[:,2,2] = sn_resample_wave(t, qij[:,2,2], fs)
    qij1[:,1,0] = qij1[:,0,1]
    qij1[:,2,0] = qij1[:,0,2]
    qij1[:,2,1] = qij1[:,1,2]    

    return t1, qij1