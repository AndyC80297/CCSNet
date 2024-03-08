
import torch

import numpy as np

from ml4gw import gw
from ml4gw.transforms import SnrRescaler
from ml4gw.utils.slicing import sample_kernels
from ml4gw.distributions import Cosine, Uniform


from ccsnet.waveform import get_hp_hc_from_q2ij, padding

class Test_Injector:


    def __init__(
        self,
        ifos: list,
        sample_rate: int,
        sample_duration,
        highpass,
        psds,
        fftlength,
        overlap,
        count,
        snr_distro,
        buffer_duration=3,
        time_shift=0,
        off_set=0,
        # save_dir
    ):
        
        
        self.sample_rate = sample_rate
        self.off_set = off_set
        self.time_shift = time_shift
        self.snr_distro = snr_distro
        self.sample_duration = sample_duration
        self.buffer_duration = buffer_duration
        if off_set is not None:
            self.max_center_offset = int((buffer_duration/2 - sample_duration - off_set) * sample_rate)
        self.count = count 
        self.tensors, self.vertices = gw.get_ifo_geometry(*ifos)
        
        self.rescaler = SnrRescaler(
            num_channels=len(ifos), 
            sample_rate = self.sample_rate,
            waveform_duration = sample_duration,
            highpass = highpass,
        )
        
        dec_distro = Cosine()
        psi_distro = Uniform(0, np.pi)
        phi_distro = Uniform(0, 2 * np.pi)
        
        # Soft init values
        self.rescaler.fit(
            psds[0, :],
            psds[1, :],
            fftlength=fftlength,
            overlap=overlap,
            use_pre_cauculated_psd=True
        )

        self.ori_theta = np.random.uniform(0, np.pi, count)
        self.ori_phi = np.random.uniform(0, 2*np.pi, count)
        self.distance = 0.1 * np.ones(count)
        self.dec = dec_distro(count)
        self.psi = psi_distro(count)
        self.phi = phi_distro(count)
        
    def __call__(
        self,
        time,
        quad_moment
    ):
        
            
        hp, hc = get_hp_hc_from_q2ij(
            quad_moment,
            theta=self.ori_theta,
            phi=self.ori_phi
        )
        
        hp_hc = padding(
            time,
            hp,
            hc,
            self.distance,
            sample_kernel = self.buffer_duration,
            sample_rate = self.sample_rate,
            time_shift = self.time_shift, # Core-bounce will be at here
        )
        
        if self.buffer_duration > self.sample_duration:
        
            hp_hc = sample_kernels(
                X = torch.Tensor(hp_hc),
                kernel_size = self.sample_rate * self.sample_duration,
                max_center_offset = self.max_center_offset,
            )
        
        ht = gw.compute_observed_strain(
            self.dec,
            self.psi,
            self.phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            plus=torch.tensor(hp_hc[:,0,:]).float(),
            cross=torch.tensor(hp_hc[:,1,:]).float()
        )
        
        
        for snr in self.snr_distro:
            
            ht, target_snrs, rescale_factor = self.rescaler.forward(
                ht,
                target_snrs = snr * torch.ones(self.count)
            )
            
            print(snr, target_snrs[-5:], rescale_factor[-5:])
        
        return ht