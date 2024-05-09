import h5py
import time
import toml
import torch

import numpy as np


from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader

from ml4gw.transforms import Whiten

from ccsnet.arch import WaveNet
from ccsnet.utils import h5_thang, args_control
from ccsnet.waveform import CCSNe_Dataset


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
        test_seg,
        background_file,
        map_device="cpu",
        device ="cpu"
    ):
        
        bgh5 = h5_thang(background_file)

        attrs = bgh5.h5_attrs()

        seg_counts = int(len(bgh5.h5_keys()) / 22)
        
        if seg_counts < test_seg:
            import sys
            sys.exit(f"Assinged segmnet segment{test_seg:02d} is too large.")

        # segs_dur = np.zeros(seg_counts)
        
        # for i in range(seg_counts):

        #     segs_dur[i] = attrs[f"segments{i:02d}/start"] - attrs[f"segments{i:02d}/end"]

        # seg = f"segments{np.argsort(segs_dur)[test_seg - 1]:02}"
        
        seg = f"segments{test_seg:02}"

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

        psds = torch.tensor(bgh5.h5_data([f"{seg}/psd"])[f"{seg}/psd"]).double()
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

