import h5py
import toml
import torch
import random

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from ccsnet.utils import h5_thang
from gwpy.timeseries import TimeSeries


def get_hp_hc_from_q2ij( 
    q2ij, 
    theta: np.ndarray, 
    phi: np.ndarray
):

    '''
    The orientation of GW emition is given by theta, phi
    '''

    hp =\
        q2ij[:,0,0]*(np.cos(theta)**2*np.cos(phi)**2 - np.sin(phi)**2).reshape(-1, 1)\
        + q2ij[:,1,1]*(np.cos(theta)**2*np.sin(phi)**2 - np.cos(phi)**2).reshape(-1, 1)\
        + q2ij[:,2,2]*(np.sin(theta)**2).reshape(-1, 1)\
        + q2ij[:,0,1]*(np.cos(theta)**2*np.sin(2*phi) - np.sin(2*phi)).reshape(-1, 1)\
        - q2ij[:,1,2]*(np.sin(2*theta)*np.sin(phi)).reshape(-1, 1)\
        - q2ij[:,2,0]*(np.sin(2*theta)*np.cos(phi)).reshape(-1, 1)

    hc = 2*(
        - q2ij[:,0,0]*(np.cos(theta)*np.sin(phi)*np.cos(phi)).reshape(-1, 1)
        + q2ij[:,1,1]*(np.cos(theta)*np.sin(phi)*np.cos(phi)).reshape(-1, 1)
        + q2ij[:,0,1]*(np.cos(theta)*np.cos(2*phi)).reshape(-1, 1)
        - q2ij[:,1,2]*(np.sin(theta)*np.cos(phi)).reshape(-1, 1)
        + q2ij[:,2,0]*(np.sin(theta)*np.sin(phi)).reshape(-1, 1)
    )

    return hp, hc

def pol_from_quad(
    quads,
    theta,
    phi
):
    """Reference paper
    https://academic.oup.com/ptps/article/doi/10.1143/PTPS.128.183/1930275
    Taking the Second-derivative quadrupole moment of a GW simulation and produce 
    polarizations.
    
    Args:
        quads (_type_): Second-derivative quadrupole moment of a GW wave
        theta (_type_): The theta angle
        phi (_type_): The phi angle

    Returns:
        THe two h_cross, h_plus polarizations
    """    

    ori_matrix_c = torch.zeros([len(theta), 3, 3])
    ori_matrix_p = torch.zeros([len(theta), 3, 3])
    
    # ori_matrix_c
    ori_matrix_c[:, 0, 0] = -2*(torch.cos(theta)*torch.sin(phi)*torch.cos(phi))
    ori_matrix_c[:, 0, 1] = 2*torch.cos(theta)*torch.cos(2*phi)
    # ori_matrix_c[:, 0, 2]
    # ori_matrix_c[:, 1, 0]
    ori_matrix_c[:, 1, 1] = 2*torch.cos(theta)*torch.sin(phi)*torch.cos(phi)
    ori_matrix_c[:, 1, 2]= -2*(torch.sin(theta)*torch.cos(phi))
    ori_matrix_c[:, 2, 0] = 2*torch.sin(theta)*torch.sin(phi)
    # ori_matrix_c[:, 2, 1]
    # ori_matrix_c[:, 2, 2]
    
    # ori_matrix_p
    ori_matrix_p[:, 0, 0] = torch.cos(theta)**2*torch.cos(phi)**2 - torch.sin(phi)**2
    ori_matrix_p[:, 0, 1] = torch.cos(theta)**2*torch.sin(2*phi) - torch.sin(2*phi)
    # ori_matrix_p[:, 0, 0]
    # ori_matrix_p[:, 1, 0]
    ori_matrix_p[:, 1, 1] = torch.cos(theta)**2*torch.sin(phi)**2 - torch.cos(phi)**2
    ori_matrix_p[:, 1, 2] = -(torch.sin(2*theta)*torch.sin(phi))
    ori_matrix_p[:, 2, 0] = -(torch.sin(2*theta)*torch.cos(phi))
    # ori_matrix_p[:, 2, 1]
    ori_matrix_p[:, 2, 2] = torch.sin(theta)**2
    
    
    return torch.einsum('kji,nij->nk', quads, ori_matrix_c), torch.einsum('kji,nij->nk', quads, ori_matrix_p)

def load_h5_as_dict(
    chosen_signals: Path,
    source_file: Path
)-> dict:
    """Open up a buffer to load in different CCSN wavefroms.

    Args:
        chosen_signals (Path): A file with names of each wavefrom.
        source_file (Path): The path that contains reasmpled raw waveform.

    Returns:
        dict: Time and resampled SQDM of Each waveform
    """
    selected_ccsn = toml.load(chosen_signals)
    source_file = Path(source_file)
    
    grand_dict = {}
    ccsn_list = []
    
    for key in selected_ccsn.keys():
        
        for name in selected_ccsn[key]:
            ccsn_list.append(f"{key}/{name}")

    for name in ccsn_list:
        
        with h5py.File(source_file/ f'{name}.h5', 'r', locking=False) as h:

            time = np.array(h['time'][:])
            quad_moment = h['quad_moment'][:] 
            
        grand_dict[name] =  [time, quad_moment]
    
    return grand_dict


def on_grid_pol_to_sim(quad_moment, sqrtnum):

    CosTheta = np.linspace(-1, 1, sqrtnum)
    theta = np.arccos(CosTheta)
    phi = np.linspace(0, 2*np.pi, sqrtnum)

    theta = theta.repeat(sqrtnum)
    phi = np.tile(phi, sqrtnum)

    hp, hc = get_hp_hc_from_q2ij(quad_moment, theta, phi)

    return hp, hc, theta, phi


def torch_on_grid_pol_to_sim(quad_moment, sqrtnum):

    CosTheta = np.linspace(-1, 1, sqrtnum)
    theta = np.arccos(CosTheta)
    phi = np.linspace(0, 2*np.pi, sqrtnum)

    theta = theta.repeat(sqrtnum)
    phi = np.tile(phi, sqrtnum)

    hp, hc = pol_from_quad(quad_moment, theta, phi)

    return hp, hc, theta, phi


# To do add masking window to the function
def padding(
    time,
    hp,
    hc,
    distance,
    sample_kernel = 3,
    sample_rate = 4096,
    time_shift = -0.15, # shift zero to distination time
):
    
    # Two polarization
    signal = np.zeros([hp.shape[0], 2, sample_kernel * sample_rate])

    half_kernel_idx = int(sample_kernel * sample_rate/2)
    time_shift_idx = int(time_shift * sample_rate)
    t0_idx = int(time[0] * sample_rate)

    start = half_kernel_idx + t0_idx + time_shift_idx
    end = half_kernel_idx + t0_idx + time.shape[0] + time_shift_idx

    signal[:, 0, start:end] = hp / distance.reshape(-1, 1)
    signal[:, 1, start:end] = hc / distance.reshape(-1, 1)
    
    return signal

def torch_padding(
    time,
    hp,
    hc,
    sample_kernel = 3,
    sample_rate = 4096,
    time_shift = -0.15, # shift zero to distination time
):

    half_kernel_idx = int(sample_kernel * sample_rate/2)
    time_shift_idx = int(time_shift * sample_rate)
    t0_idx = int(time[0] * sample_rate)

    start = half_kernel_idx + t0_idx + time_shift_idx
    end = sample_kernel * sample_rate - (start + time.shape[0])
    if end > 0:
        pass
    p1d = (start, end)
    hp = F.pad(hp, pad=p1d, mode="constant", value=0)
    hc = F.pad(hc, pad=p1d, mode="constant", value=0)

    return hp, hc