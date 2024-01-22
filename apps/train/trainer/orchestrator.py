import toml
import torch

import numpy as np
from tqdm import tqdm

# from ccsnet.arch import WaveNet
from ccsnet.sampling import masking, strain_sampling, glitch_sampler
from ccsnet.sampling import tapping
from ccsnet.waveform import load_h5_as_dict, get_hp_hc_from_q2ij, padding
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
        # background_duration,
        # segment_start_time,
        # max_iteration, 
        batch_size,
        steps_per_epoch,
        sample_rate,
        sample_duration,
        sample_factor = 1/2,
        
    ):
        # Sample_factor <= 1
        bg_info = h5_thang(background_file)
        background = bg_info.h5_data()["segments16/strain"]
        bg_attrs = bg_info.h5_attrs()
        background_duration = bg_attrs["segments16/end"] - bg_attrs["segments16/start"]
        
        
        self.background = torch.Tensor(background)
        self.glitch_info = glitch_info
        self.background_duration = background_duration
        self.max_iteration = steps_per_epoch
        self.segment_start_time = bg_attrs["segments16/start"]
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.ifos = ifos
        self.num_ifos = len(ifos)
        
        self.kernel_length = sample_rate * sample_duration
        self.sample_data = int(batch_size * steps_per_epoch * sample_factor)
        
    def __call__(
        self,
        glitch_dist: list,
        # sample_method = None
        choice_mask = [0, 1, 2, 3],
        glitch_offset = 0.9,
        target_value = 0
    ):
        print(__file__, "Start calling")
        X = torch.empty((self.sample_data, self.num_ifos, self.kernel_length))

        mask = np.random.choice(
            choice_mask, 
            self.sample_data, 
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
            
        targets = torch.full((self.sample_data,), target_value)
        
        return X, targets

class Injector:

    def __init__(
        self,
        ifos,
        background,
        signals_dict,
        # max_iteration, 
        # batch_size,
        # steps_per_epoch
        sample_rate,
        sample_duration,
        # num_ifos,
        # highpass
    ):


        self.tensors, self.vertices = gw.get_ifo_geometry(*ifos)
        
        self.total_counts = background.shape[0]
        self.ccsn_list = list(signals_dict.keys())
        
        ccsn_num = len(self.ccsn_list)
        ccsn_sample = np.random.choice(ccsn_num, self.total_counts)
        self.ccsn_counts = np.eye(ccsn_num)[ccsn_sample].sum(0).astype("int")
        
        self.background = background
        
        # self.max_iteration = max_iteration
        # self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        # self.num_ifos = num_ifos
        
        self.kernel_length = sample_rate * sample_duration
        # self.sample_data = int(max_iteration*batch_size/2)
        self.signals = signals_dict
        # self.highpass = highpass
        
        
    def __call__(
        self,
        max_distance = None,
    ):
    
    
        X = torch.empty((self.total_counts, 2, self.kernel_length))
        agg_count = 0

        for name, count in tqdm(zip(self.ccsn_list, self.ccsn_counts)):
            
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
            
            X[agg_count:agg_count+count, :, :] = torch.Tensor(hp_hc)
            
        # prior = PriorDict()
        dec_distro = Cosine()
        psi_distro = Uniform(0, np.pi)
        phi_distro = Uniform(0, 2 * np.pi)
        
        dec = dec_distro(self.total_counts)
        psi = psi_distro(self.total_counts)
        phi = phi_distro(self.total_counts)
        
        
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
        
        
        
        # ### We have to change this to use pre-caulated psd
        # rescaler = SnrRescaler(
        #     num_channels=self.num_ifos, 
        #     sample_rate = self.sample_rate,
        #     waveform_duration = self.sample_duration,
        #     highpass = self.highpass,
        # )


        # rescaler.fit(
        #     self.signals[0], 
        #     self.signals[1],
        #     fftlength=self.sample_duration,
        #     overlap=1,
        # )

        # rescaled_signals, target_snrs, rescale_factor = rescaler.forward(
        #     self.signals[:, :, 4096:12288]
        # )
        
        ht += self.background
        # # targets = torch.full((self.sample_data,), 1)
        return  ht
    
def forged_dataloader(
    inputs: list,
    targets: list,
    batch_size,
    pin_memory=True,
    # n_workers
):


    #### Apply whiten
    dataset = torch.utils.data.TensorDataset(
        torch.cat(inputs), 
        torch.cat(targets).view(-1, 1).to(torch.float)
    )
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)