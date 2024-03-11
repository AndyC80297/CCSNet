
import torch
import h5py

import numpy as np

from ml4gw import gw
from ml4gw.transforms import SnrRescaler
from ml4gw.utils.slicing import sample_kernels
from ml4gw.distributions import Cosine, Uniform

from ccsnet.utils import h5_thang
from ccsnet.sampling import tapping
from ccsnet.sampling import masking, strain_sampling, glitch_sampler
from ccsnet.waveform import get_hp_hc_from_q2ij, padding

class Test_BackGroundDisplay:
    
    def __init__(
        self,
        ifos,
        background,
        background_dur,
        start_time,
        glitch_info,
        sample_rate,
        sample_duration
    ):
        
        # Sample_factor <= 1
        
        self.background = background
        self.bg_dur = background_dur
        self.start_time = start_time

        
        self.glitch_info = glitch_info
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
        target_value = 0,
        batch_size = 32,
        steps_per_epoch = 20,
        sample_factor = 1/2 # This will have to be returned to check if the total sample 
        # factor equals to one.
    ):

        num_sample_data = int(batch_size * steps_per_epoch * sample_factor)
        
        X = torch.empty((num_sample_data, self.num_ifos, self.kernel_length))

        mask = np.random.choice(
            choice_mask, 
            num_sample_data, 
            p=glitch_dist
        )
        
        glitch_tape = tapping(self.num_ifos, mask)
        
        for i, ifo in enumerate(self.ifos):

            glitch_label = h5_thang(self.glitch_info).h5_data([f"{ifo}/time"])

            glitch_mask = glitch_tape[i, :].astype("bool")
            glitch_count = glitch_mask.sum()
            

            X[glitch_mask, i, :] = glitch_sampler(
            # gltich_data_ =  glitch_sampler(
                gltich_info=glitch_label,
                strain = self.background[i, :],
                segment_duration = self.bg_dur,
                segment_start_time = self.start_time,
                ifos = [ifo],
                sample_counts = glitch_count,
                sample_rate = self.sample_rate,
                shift_range = glitch_offset,
                kernel_width = self.sample_duration,
            )
            
            # print(type(gltich_data_))
            
            reverse_mask = (1 - glitch_tape[i, :]).astype("bool")
            
            reverse_count = reverse_mask.sum()
            glitch_label = h5_thang(self.glitch_info).h5_data([f"{ifo}/time"])
            
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
            
        targets = torch.full((num_sample_data,), target_value)
        
        # with h5py.File(self.outdir / "background.h5", "a") as g:
            
        #     g.create_dataset(f"{iteration:03d}_{mode}", data=X.numpy())
        
        return X, targets

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
        save_dir=None,
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
        
        if save_dir is not None:
            
            with h5py.File(save_dir / "parameters.h5", "w") as g:
                
                g.create_dataset("ori_theta", data=self.ori_theta)
                g.create_dataset("ori_phi", data=self.ori_phi)
                g.create_dataset("dec", data=self.dec)
                g.create_dataset("psi", data=self.psi)
                g.create_dataset("phi", data=self.phi)
                
                
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
            # plus=torch.tensor(hp_hc[:,0,:]).float(),
            # cross=torch.tensor(hp_hc[:,1,:]).float(),
            plus=hp_hc[:,0,:].clone().detach(),
            cross=hp_hc[:,1,:].clone().detach()
        )
        
        
        for snr in self.snr_distro:
            
            scaled_ht, _, rescale_factor = self.rescaler.forward(
                ht,
                target_snrs = snr * torch.ones(self.count)
            )
            
            
            yield snr, scaled_ht, rescale_factor