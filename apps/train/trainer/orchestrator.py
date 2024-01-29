import toml
import torch

import numpy as np
from tqdm import tqdm

# from ccsnet.arch import WaveNet
from ccsnet.sampling import masking, strain_sampling, glitch_sampler
from ccsnet.sampling import tapping
from ccsnet.waveform import get_hp_hc_from_q2ij, padding
from ccsnet.utils import h5_thang

from ml4gw.transforms import SnrRescaler
from ml4gw.distributions import PowerLaw, Cosine, Uniform
from ml4gw import gw


class BackGroundDisplay:
    
    def __init__(
        self,
        ifos,
        background_file,
        glitch_info,
        sample_rate,
        sample_duration,
    ):
        
        # Sample_factor <= 1
        bg_info = h5_thang(background_file)

        background = bg_info.h5_data()["segments00/strain"]
        bg_attrs = bg_info.h5_attrs()
        background_duration = bg_attrs["segments00/end"] - bg_attrs["segments00/start"]
        
        
        self.background = torch.Tensor(background)
        self.glitch_info = glitch_info
        self.background_duration = background_duration
        self.segment_start_time = bg_attrs["segments00/start"]
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
                gltich_info=glitch_label,
                strain = self.background[i, :],
                segment_duration = self.background_duration,
                segment_start_time = self.segment_start_time,
                ifos = [ifo],
                sample_counts = glitch_count,
                sample_rate = self.sample_rate,
                shift_range = glitch_offset,
                kernel_width = self.sample_duration,
            )
            
            
            reverse_mask = (1 - glitch_tape[i, :]).astype("bool")
            
            reverse_count = reverse_mask.sum()
            glitch_label = h5_thang(self.glitch_info).h5_data([f"{ifo}/time"])

            mask_dict = masking(
                glitch_info=glitch_label,
                segment_duration=self.background_duration,
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
        
        return X, targets

class Injector:

    def __init__(
        self,
        ifos,
        # background,
        signals_dict,
        sample_rate,
        sample_duration,
    ):

        self.tensors, self.vertices = gw.get_ifo_geometry(*ifos)
        
        self.signals = signals_dict
        self.ccsn_list = list(signals_dict.keys())
        
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.kernel_length = sample_rate * sample_duration
        
    def __call__(
        self,
        background,
        max_distance = None,
    ):
    
        total_counts = background.shape[0]
        
        ccsn_num = len(self.ccsn_list)
        ccsn_sample = np.random.choice(ccsn_num, total_counts)
        ccsn_counts = np.eye(ccsn_num)[ccsn_sample].sum(0).astype("int")
        
        X = torch.empty((total_counts, 2, self.kernel_length))
        agg_count = 0

        for name, count in tqdm(zip(self.ccsn_list, ccsn_counts), total=len(self.ccsn_list)):
            
            time = self.signals[name][0]
            quad_moment = self.signals[name][1]
        

            theta = np.random.uniform(0, np.pi, count)
            phi = np.random.uniform(0, 2*np.pi, count)            
            
            dist_distro = PowerLaw(1, max_distance[name], alpha=3)
            
            distance = 0.1*dist_distro(count)
            
            hp, hc = get_hp_hc_from_q2ij(
                quad_moment,
                theta=theta,
                phi=phi
            )

            hp_hc = padding(
                time,
                hp,
                hc,
                distance.numpy(),
                sample_kernel = self.sample_duration,
                sample_rate = self.sample_rate,
                time_shift = -0.15, # Core-bounce will be at here
            )
            
            # To Do:
            # Add ml4gw.sample_kerel function here to shift the CCSN signal
            
            X[agg_count:agg_count+count, :, :] = torch.Tensor(hp_hc)
            
        # prior = PriorDict()
        dec_distro = Cosine()
        psi_distro = Uniform(0, np.pi)
        phi_distro = Uniform(0, 2 * np.pi)
        
        dec = dec_distro(total_counts)
        psi = psi_distro(total_counts)
        phi = phi_distro(total_counts)
        
        ht = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            plus=X[:,1,:],
            cross=X[:,1,:]
        )
        
        background += ht

        return  background
    
    
def forged_dataloader(
    inputs: list,
    targets: list,
    batch_size,
    # whiten_model,
    # psd,
    pin_memory=True,
    # n_workers
):

    #### Apply whiten
    dataset = torch.utils.data.TensorDataset(
        torch.cat(inputs).to("cuda"), 
        torch.cat(targets).view(-1, 1).to(torch.float).to("cuda")
    )
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)