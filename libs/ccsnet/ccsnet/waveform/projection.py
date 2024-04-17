import time as ti
import torch

import numpy as np

from pathlib import Path

from ml4gw import gw
from ml4gw.transforms import SnrRescaler
from ml4gw.utils.slicing import sample_kernels

from ccsnet.utils import h5_thang
from ccsnet.waveform import get_hp_hc_from_q2ij, padding
from ccsnet.waveform import pol_from_quad, torch_padding


class Waveform_Projector: 

    def __init__(
        self,
        ifos,
        sample_rate,
        background_file,
        seg,
        highpass,
        fftlength,
        overlap,
        sample_duration,
        buffer_duration=3,
        time_shift=0,
        off_set=0,
    ):  
        """_summary_

        Args:
            ifos (_type_): _description_
            sample_rate (_type_): _description_
            background_file (_type_): For the psd.
            seg (_type_): _description_
            highpass (_type_): _description_
            fftlength (_type_): _description_
            overlap (_type_): _description_
            sample_duration (_type_): _description_
            buffer_duration (int, optional): _description_. Defaults to 3.
            time_shift (int, optional): Shifts Core-bounce to "Time Shift". Defaults to 0.
            off_set (int, optional): _description_. Defaults to 0.
        """


        self.ifos = ifos
        self.sample_rate = sample_rate

        bgh5 = h5_thang(background_file)
        psds = torch.tensor(bgh5.h5_data([f"{seg}/psd"])[f"{seg}/psd"]).double()
        
        self.tensors, self.vertices = gw.get_ifo_geometry(*self.ifos)

        self.sample_duration = sample_duration
        self.rescaler = SnrRescaler(
            num_channels=len(self.ifos), 
            sample_rate = self.sample_rate,
            waveform_duration = self.sample_duration,
            highpass = highpass,
        )
        
        self.rescaler.fit(
            psds[0, :],
            psds[1, :],
            fftlength=fftlength,
            overlap=overlap,
            use_pre_cauculated_psd=True
        )

        self.buffer_duration = buffer_duration

        self.off_set = off_set
        self.time_shift = time_shift

        if off_set is not None:
            
            self.max_center_offset = int((buffer_duration/2 - sample_duration - off_set) * sample_rate)

    def __call__(
        self,
        time,
        quad_moment,
        ori_theta,
        ori_phi,
        dec,
        psi,
        phi,
        default_snr: float = 4
    ):
        
        count = len(ori_theta)
        
        hp, hc = get_hp_hc_from_q2ij(
            quad_moment,
            theta=ori_theta,
            phi=ori_phi
        )
        
        hp_hc = padding(
            time,
            hp,
            hc,
            np.ones(count),
            sample_kernel = self.buffer_duration,
            sample_rate = self.sample_rate,
            time_shift = self.time_shift, # Core-bounce will be at here
        )

        hp_hc = torch.tensor(hp_hc).float()
        
        if self.buffer_duration > self.sample_duration:
        
            hp_hc = sample_kernels(
                X = hp_hc,  
                kernel_size = self.sample_rate * self.sample_duration,
                max_center_offset = self.max_center_offset,
            )

        ht = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            plus=hp_hc[:,0,:],
            cross=hp_hc[:,1,:]
        )
        
        if default_snr is None:

            return scaled_ht
        if default_snr is not None:
            scaled_ht, _, inversed_distance = self.rescaler.forward(
                ht,
                target_snrs = default_snr * torch.ones(count)
            )
            
            return scaled_ht, 1/inversed_distance