import sys
import h5py
import torch

import numpy as np
import logging

from pathlib import Path

from ml4gw import gw
from ml4gw.transforms import SnrRescaler
from ml4gw.utils.slicing import sample_kernels
from ml4gw.distributions import Cosine, Uniform

from ccsnet.utils import h5_thang
from ccsnet.sampling import tapping
from ccsnet.sampling import masking, strain_sampling, glitch_sampler
from ccsnet.omicron import glitch_merger, psd_estimiater
from ccsnet.waveform import get_hp_hc_from_q2ij, padding

class Test_BackGroundDisplay:
    
    def __init__(
        self,
        ifos: list,
        background_file : Path, 
        sample_duration: float,
        sample_rate: int = 4096,
        test_seg: int = 1
    ):
        """Load background data and glitch trigger times to memmory.

        Args:
            ifos (list): Detectors.
            background_file (Path): Path of background to load.
            glitch_info_file (Path): Path of glitch to load.
            sample_duration (float): Time window(sec) to sample background.
            sample_rate (int, optional): Sampling rate of the strain data. Defaults to 4096.
            test_seg (int, optional): The nth largest activae segment . Defaults to 1.
        """
        
        bgh5 = h5_thang(background_file)

        attrs = bgh5.h5_attrs()

        seg_counts = int(len(bgh5.h5_keys()) / 22)
        
        if seg_counts < test_seg:
            sys.exit(f"Assinged segmnet segment{test_seg:02d} is too large.")

        # segs_dur = np.zeros(seg_counts)
        
        # for i in range(seg_counts):

        #     segs_dur[i] = attrs[f"segments{i:02d}/start"] - attrs[f"segments{i:02d}/end"]

        # seg = f"segments{np.argsort(segs_dur)[test_seg - 1]:02}"

        seg = f"segments{test_seg:02}"
        
        start = attrs[f"{seg}/start"]
        end = attrs[f"{seg}/end"]
        
        self.background = torch.Tensor(bgh5.h5_data([f"{seg}/strain"])[f"{seg}/strain"])
        self.bgh5 = bgh5
        self.seg = seg
        self.bg_dur = end - start
        self.start_time = start
        
        logging.info(f"Analysing sement {seg}.")
        logging.info(f"  Start time: {int(start)}")
        logging.info(f"  -End- time: {int(end)}")
        logging.info(f"Duration: {int(self.bg_dur)}")

        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.ifos = ifos
        self.num_ifos = len(ifos)
        
        self.kernel_length = sample_rate * sample_duration
        
    def __call__(
        self,
        glitch_dist: list,
        choice_mask = [0, 1, 2, 3],
        glitch_offset = 0.9,
        test_count: int = 5000
    ):
        
        num_sample_data = test_count
        
        X = torch.empty((num_sample_data, self.num_ifos, self.kernel_length))

        mask = np.random.choice(
            choice_mask, 
            num_sample_data, 
            p=glitch_dist
        )
        
        glitch_tape = tapping(self.num_ifos, mask)
        glitch_label = {}
        for i, ifo in enumerate(self.ifos):

            glitch_label[f"{ifo}/time"] = self.bgh5.h5_data([f"{self.seg}/{ifo}/time"])[f"{self.seg}/{ifo}/time"]

            glitch_mask = glitch_tape[i, :].astype("bool")
            glitch_count = glitch_mask.sum()
            
            X[glitch_mask, i, :] = glitch_sampler(
                glitch_info=glitch_label,
                strain = self.background[i, :],
                segment_duration = self.bg_dur,
                segment_start_time = self.start_time,
                ifos = [ifo],
                sample_counts = glitch_count,
                sample_rate = self.sample_rate,
                shift_range = glitch_offset,
                kernel_width = self.sample_duration,
            )
            
            reverse_mask = (1 - glitch_tape[i, :]).astype("bool")
            
            reverse_count = reverse_mask.sum()
            
            mask_dict = masking(
                glitch_info=glitch_label,
                segment_duration=self.bg_dur,
                segment_start_time=self.start_time,
                shift_range=self.sample_duration,
                pad_width=self.sample_duration/2,
                merge_edges = True
            )

            X[reverse_mask, i, :] = strain_sampling(
                self.background[i, :],
                mask_dict,
                sample_counts=reverse_count,
                kernel_width=self.sample_duration
            )
        
        return X
    
    def get_psds(
        self,
        fftlength,
        overlap,
        device="cpu"
    ):

        psds = psd_estimiater(
            ifos=self.ifos,
            strains=torch.tensor(self.background).double(),  # This part need to modify to segments-wise operation
            sample_rate=self.sample_duration,
            kernel_width=self.sample_duration,
            fftlength=fftlength,
            overlap=overlap
        ).to(device)
        

        return psds.to(device)

def random_ccsn_parameter(
    count:int ,
    save_path: Path = None 
):
    
    dec_distro = Cosine()
    psi_distro = Uniform(0, np.pi)
    phi_distro = Uniform(0, 2 * np.pi)

    ori_theta = np.random.uniform(0, np.pi, count)
    ori_phi = np.random.uniform(0, 2*np.pi, count)
    dec = dec_distro(count)
    psi = psi_distro(count)
    phi = phi_distro(count)
    
    if save_path.is_file():

        return ori_theta, ori_phi, dec, psi, phi
    
    with h5py.File(save_path, "a") as g:
        
        g.create_dataset("ori_theta", data=ori_theta)
        g.create_dataset("ori_phi", data=ori_phi)
        g.create_dataset("dec", data=dec)
        g.create_dataset("psi", data=psi)
        g.create_dataset("phi", data=phi)
            
    return ori_theta, ori_phi, dec, psi, phi


