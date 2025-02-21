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

def replacement(
    data,
    coh_det,
    roll:int=0,
):

    ndim = len(data.shape)

    if ndim == 2:
        
        if coh_det == "H1":
            data[1,:] = torch.roll(data[0,:], roll, dims=-1)
        
        elif coh_det == "L1":
            data[0,:] = torch.roll(data[1,:], roll, dims=-1)
            
        else:
            logging.info("Unknow ifo specifed!")
            
    elif ndim == 3:

        if coh_det == "H1":
            data[:,1,:] = torch.roll(data[:,0,:], roll, dims=-1)
        
        elif coh_det == "L1":
            data[:,0,:] = torch.roll(data[:,1,:], roll, dims=-1)
            
        else:
            logging.info("Unknow ifo specifed!")

    else:
        logging.info("Dimension of input unclear.")
        
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
        coh_det=None,
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
                
        if coh_det is not None:
            psds = replacement(
                data=psds,
                coh_det=coh_det,
            )

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
        X = self.nn_model(X)
        
        return X

