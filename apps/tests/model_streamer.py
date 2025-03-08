import h5py
import time
import toml
import torch
import logging

import numpy as np


from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader

from ml4gw.transforms import Whiten

from ccsnet.arch import WaveNet
from ccsnet.utils import h5_thang, args_control
from ccsnet.waveform import CCSNe_Dataset

def strain_state(
    data,
    coh_ifo=None,
    coh_mode=None,
    roll:int=0,
):

    if coh_mode is None:
        
        return data
    
    ndim = len(data.shape)

    if coh_mode == "replace":
        # breakpoint()
        if ndim == 2:
            
            if coh_ifo == "H1":
                data[1,:] = torch.roll(data[0,:], roll, dims=-1)
            
            elif coh_ifo == "L1":
                data[0,:] = torch.roll(data[1,:], roll, dims=-1)
                
            else:
                logging.info("Unknow ifo specifed!")
                
        elif ndim == 3:

            if coh_ifo == "H1":
                data[:,1,:] = torch.roll(data[:,0,:], roll, dims=-1)
            
            elif coh_ifo == "L1":
                data[:,0,:] = torch.roll(data[:,1,:], roll, dims=-1)
                
            else:
                logging.info("Unknow ifo specifed!")

        else:
            logging.info("Dimension of input unclear.")
            
        return data

    if coh_mode == "remove":
        # breakpoint()
        if ndim == 2:
            # PSD no need of remove
            if coh_ifo == "H1":
                data[1,:] = data[0,:]
            
            elif coh_ifo == "L1":
                data[0,:] = data[1,:]
                
            else:
                logging.info("Unknow ifo specifed!")
                
        elif ndim == 3:
            kernel_size = data.size(-1)
            if coh_ifo == "H1":

                data = torch.roll(data[:,0,:], roll, dims=-1)

            elif coh_ifo == "L1":
                data = torch.roll(data[:,1,:], roll, dims=-1)
                
            else:
                logging.info("Unknow ifo specifed!")
            data = data.reshape((-1, 1, kernel_size))
        return data
                

def model_loader(
    num_ifos: int,
    architecture,
    model_weights,
    map_device = "cpu",
    device="cpu"
):
    
    nn_model = architecture(num_ifos)

    state_dict = torch.load(
        model_weights, 
        map_location=torch.device(map_device), 
        weights_only=True
    )
    
    nn_model.load_state_dict(state_dict)
    nn_model.eval()
    
    nn_model.to(device)
    
    return nn_model


def test_data_loader(
    signal,
    n_ifos=2, 
    sample_rate=4096,
    sample_duration=3,
    batch_size: int = 1024,
    shuffle=False,
    device="cpu",
    scaled_distance=None,
):

    dataset = CCSNe_Dataset(
        signal,
        scaled_distance,
        n_ifos=n_ifos, 
        sample_rate=sample_rate,
        sample_duration=sample_duration,
        device=device
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class Streamer:

    def __init__(
        self,
        num_ifos: int,
        sample_rate: int,
        architecture,
        model_weights,
        fftlength,
        highpass,
        test_psd_seg,
        coh_ifo=None,
        coh_mode=None,
        map_device="cpu",
        device ="cpu"
    ):
        
        self.nn_model = model_loader(
            num_ifos=num_ifos,
            architecture = architecture,
            model_weights=model_weights,
            map_device=map_device,
            device = device
        )

        self.whiten_model = Whiten(
            fftlength,
            sample_rate,
            highpass,
        ).to(device)

        if test_psd_seg is not None:

            with h5py.File(test_psd_seg, "r") as h1:
                psds = torch.tensor(h1["psd"][:]).double()
                
        self.coh_ifo = coh_ifo
        self.coh_mode = coh_mode  
        
        self.psds = psds.to(device)
        self.device = device
        
    def __call__(
        self,
        dataloader,
    ):
        
        with torch.no_grad():

            for signal, index in dataloader:
                
                signal = self.whiten_model(
                    signal,
                    self.psds
                )

                signal = strain_state(
                    data=signal,
                    coh_ifo=self.coh_ifo,
                    coh_mode=self.coh_mode
                )
                breakpoint()
                pred = self.nn_model(signal)
                
                yield pred, index
            
    def stream(
        self,
        X,
        psd = None
    ):
        
        if psd is not None:
            psds = psd.to(self.device)
        else:
            psds = self.psds
            
        X = self.whiten_model(X, psds)
        X = strain_state(
            data=X,
            coh_ifo=self.coh_ifo,
            coh_mode=self.coh_mode
        )
        X = self.nn_model(X)
        
        return X

