import h5py
import toml
import random

import numpy as np

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


def load_h5_as_dict(ccsn_dict):

    grand_dict = {}
    # ccsn_data_base = Path.home() / 'anti_gravity/CCSNet/libs/data_base/'
    ccsn_data_base = Path.home() / 'Data/3DCCSN_PREMIERE/Resampled'
    
    for family in ccsn_dict.keys():

        data_dict = {}
        for signal in ccsn_dict[family]:

            with h5py.File(ccsn_data_base/ f'{family}/{signal}.h5', 'r', locking=False) as h1:

                time = np.array(h1['time'][:])
                quad_moment = h1['quad_moment'][:]

            data_dict[signal] = [time, quad_moment]

        grand_dict[family] = data_dict

    return grand_dict


def pol_from_sim(grand_dict, family, signal, sqrtnum):

    time = grand_dict[family][signal][0]
    quad_moment = grand_dict[family][signal][1]

    # theta = np.random.uniform(0, np.pi, 1)
    # phi = np.random.uniform(0, 2*np.pi, 1)

    phi = np.linspace(0, 2*np.pi, sqrtnum)
    CosTheta = np.linspace(-1, 1, sqrtnum)
    theta = np.arccos(CosTheta)

    hp, hc = get_hp_hc_from_q2ij(quad_moment, theta, phi)

    return time, hp, hc, theta, phi


def padding(
    signal,
    time,
    dur=8,
    time_shift=-0.15, # shift zero to distination time
    sample_rate=4096
):

    data = TimeSeries(
        signal,
        sample_rate=sample_rate,
        t0=int((time[0] + time_shift)*sample_rate)/sample_rate
    )

    blank = TimeSeries(
        np.zeros(dur*sample_rate),
        sample_rate=sample_rate,
        t0=-dur/2
    )

    return blank.inject(data)


def padding_2(
    signal,
    time,
    dur=8,
    time_shift=-0.15, # shift zero to distination time
    sample_rate=4096
):

    data = TimeSeries(
        signal,
        sample_rate=sample_rate,
        t0=int((time[0] + time_shift)*sample_rate)/sample_rate
    )

    blank = TimeSeries(
        np.zeros(dur*sample_rate),
        sample_rate=sample_rate,
        t0=-dur/2
    )
    
    the_ones = TimeSeries(
        np.ones(16588),
        sample_rate=4096,
        t0=-int(.05*sample_rate)/sample_rate
    )

    the_ones = TimeSeries(
        np.ones(dur*sample_rate),
        sample_rate=sample_rate,
        t0=-dur/2
    )

    augmentator = np.append(np.ones(16999), np.zeros(15769))
    
    augmentator = TimeSeries(
        augmentator,
        sample_rate=4096,
        t0=-dur/2
    )
    
    data = blank.inject(data)
    # augmentator = blank.inject(the_ones)
    
    return data*augmentator
