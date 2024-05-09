import h5py
import toml
import torch
import logging

import numpy as np
from tqdm import tqdm

from ccsnet.sampling import masking, strain_sampling, glitch_sampler
from ccsnet.sampling import tapping
from ccsnet.waveform import get_hp_hc_from_q2ij, padding, pol_from_quad, torch_padding
from ccsnet.utils import h5_thang


from ml4gw.transforms import SnrRescaler
from ml4gw.utils.slicing import sample_kernels
from ml4gw.distributions import PowerLaw, Cosine, Uniform
from ml4gw import gw

class SignalInverter(torch.nn.Module):
    """
    Takes a tensor of timeseries of arbitrary dimension
    and randomly inverts (i.e. h(t) -> -h(t))
    each timeseries with probability `prob`.

    Args:
        prob:
            Probability that a timeseries is inverted
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, X):
        mask = torch.rand(size=X.shape[:-1]) < self.prob
        X[mask] *= -1
        return X


class SignalReverser(torch.nn.Module):
    """
    Takes a tensor of timeseries of arbitrary dimension
    and randomly reverses (i.e. h(t) -> h(-t))
    each timeseries with probability `prob`.

    Args:
        prob:
            Probability that a kernel is reversed
    """

    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, X):
        mask = torch.rand(size=X.shape[:-1]) < self.prob
        X[mask] = X[mask].flip(-1)
        return X

class BackGroundDisplay:
    
    def __init__(
        self,
        ifos,
        background_file,
        segment,
        sample_rate,
        sample_duration,
        outdir,
        training_portion = 0.75
    ):
        
        # Sample_factor <= 1
        self.segment = segment
        self.bg_dur = {}
        self.start_time = {}
        self.background = {}
        

        self.bgh5 = h5_thang(background_file)
        background = torch.Tensor(self.bgh5.h5_data([f"{segment}/strain"])[f"{segment}/strain"])
        bg_attrs = self.bgh5.h5_attrs()
        bg_dur = bg_attrs[f"{segment}/end"] - bg_attrs[f"{segment}/start"]
        self.train_dur = int(bg_dur * training_portion)

        self.bg_dur["Train"] = int(bg_dur * training_portion)
        self.bg_dur["Validate"] = int(bg_dur * (1 - training_portion))

        self.start_time["Train"] = bg_attrs[f"{segment}/start"]
        self.start_time["Validate"] = bg_attrs[f"{segment}/start"] + self.bg_dur["Train"]

        self.background["Train"] = background[:,:self.bg_dur["Train"]*sample_rate]
        self.background["Validate"] = background[:,-self.bg_dur["Validate"]*sample_rate:]

        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.ifos = ifos
        self.num_ifos = len(ifos)

        glitch_keys = []
        for ifo in self.ifos:

            glitch_keys.append(f"{self.segment}/{ifo}/time")
            
        
        glitch_label = self.bgh5.h5_data(glitch_keys)

        for ifo in self.ifos:
            
            glitch_label[f"{ifo}/time"] = glitch_label.pop(f"{self.segment}/{ifo}/time")
        
        self.glitch_label = glitch_label
        self.kernel_length = sample_rate * sample_duration
        self.outdir = outdir / "raw_data"
        self.outdir.mkdir(parents=True, exist_ok=True)
        
    def __call__(
        self,
        glitch_dist: list,
        mode,
        choice_mask = [0, 1, 2, 3],
        glitch_offset = 0.9,
        target_value = 0,
        batch_size = 32,
        steps_per_epoch = 20,
        sample_factor = 1/2,  # This will have to be returned to check if the total sample 
        # factor equals to one.
        iteration=None,
        noise_mode=None
    ):

        num_sample_data = int(batch_size * steps_per_epoch * sample_factor)
        
        X = torch.empty((num_sample_data, self.num_ifos, self.kernel_length))

        if glitch_dist is not None:

            mask = np.random.choice(
                choice_mask, 
                num_sample_data, 
                p=glitch_dist
            )

            glitch_tape = tapping(self.num_ifos, mask)
            for i, ifo in enumerate(self.ifos):

                glitch_mask = glitch_tape[i, :].astype("bool")

                glitch_count = glitch_mask.sum()
                
                
                X[glitch_mask, i, :] = glitch_sampler(
                    glitch_info=self.glitch_label,
                    strain = self.background[mode][i, :],
                    segment_duration = self.bg_dur[mode],
                    segment_start_time = self.start_time[mode],
                    ifos = [ifo],
                    sample_counts = glitch_count,
                    sample_rate = self.sample_rate,
                    shift_range = glitch_offset,
                    kernel_width = self.sample_duration,
                )
                
                reverse_mask = (1 - glitch_tape[i, :]).astype("bool")
                reverse_count = reverse_mask.sum()

                mask_dict = masking(
                    glitch_info=self.glitch_label,
                    segment_duration=self.bg_dur[mode],
                    segment_start_time=self.start_time[mode],
                    shift_range=self.sample_duration,
                    pad_width=self.sample_duration/2,
                    merge_edges = True
                )
                
                X[reverse_mask, i, :] = strain_sampling(
                    self.background[mode][i, :],
                    mask_dict,
                    sample_counts=reverse_count,
                    kernel_width=self.sample_duration
                )

        else:

            for i, ifo in enumerate(self.ifos):

                mask_dict = masking(
                    glitch_info=self.glitch_label,
                    segment_duration=self.bg_dur[mode],
                    segment_start_time=self.start_time[mode],
                    shift_range=self.sample_duration,
                    pad_width=self.sample_duration/2,
                    merge_edges = True,
                    edge_only=True
                )
                
                X[:, i, :] = strain_sampling(
                    self.background[mode][i, :],
                    mask_dict,
                    sample_counts=num_sample_data,
                    kernel_width=self.sample_duration
                )

        targets = torch.full((num_sample_data,), target_value)
        
        bg_mode = {
            0: "Noise",
            1: "Injection"
        }
        if iteration is not None:
            
            data_name=f"{iteration:03d}_{mode}_{bg_mode[target_value]}"

            if noise_mode is not None:

                data_name += f"_{noise_mode}"

            with h5py.File(self.outdir / "background.h5", "a") as g:
                
                print(f"Eatting popcornes! At {data_name}")
                g.create_dataset(data_name, data=X.numpy())
        
        return X, targets

class Injector:

    def __init__(
        self,
        ifos,
        # background,
        signals_dict,
        sample_rate,
        sample_duration,
        # init_distance,
        psds,
        fftlength,
        overlap, 
        outdir,
        batch_size = 32,
        steps_per_epoch = 20,
        buffer_duration = 4,
        off_set = 0.15,
        time_shift = -0.05
    ):

        self.tensors, self.vertices = gw.get_ifo_geometry(*ifos)
        
        self.signals = signals_dict
        self.ccsn_list = list(signals_dict.keys())
        
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.buffer_duration = buffer_duration
        self.off_set = off_set
        self.time_shift = time_shift
        self.kernel_length = sample_duration * sample_rate
        self.buffer_length = buffer_duration * sample_rate
        self.max_center_offset = int((buffer_duration/2 - sample_duration - off_set) * sample_rate)
        self.outdir = outdir / "raw_data"
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        self.snr_distro = PowerLaw(12, 100, 3)
        
        self.rescaler = SnrRescaler(
            num_channels=len(ifos), 
            sample_rate = sample_rate,
            waveform_duration = sample_duration,
            highpass = 32,
        )

        self.rescaler.fit(
            psds[0, :],
            psds[1, :],
            fftlength=fftlength,
            overlap=overlap,
            use_pre_cauculated_psd=True
        )


        if off_set <= -time_shift:
            
            logging.info(f"Core bounce siganl may leak out of sample kernel by {-time_shift - off_set}")
        
    def __call__(
        self,
        background,
        iteration=None,
    ):
    
        total_counts = background.shape[0]
        
        ccsn_num = len(self.ccsn_list)
        ccsn_sample = np.random.choice(ccsn_num, total_counts)
        ccsn_counts = np.eye(ccsn_num)[ccsn_sample].sum(0).astype("int")
        
        X = torch.empty((total_counts, 2, self.kernel_length))
        agg_count = 0

        for name, count in zip(self.ccsn_list, ccsn_counts):    
        
            time = self.signals[name][0]
            quad_moment = torch.Tensor(self.signals[name][1] * 0.1) # 0.1 is to rescale the distance to 1kpc
            
            
            theta = torch.Tensor(np.random.uniform(0, np.pi, count))
            phi = torch.Tensor(np.random.uniform(0, 2*np.pi, count))       

            hp, hc = get_hp_hc_from_q2ij(
                quad_moment,
                theta=theta,
                phi=phi
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
            
            shifted_waveforms = sample_kernels(
                X = torch.Tensor(hp_hc),
                kernel_size = self.sample_rate * self.sample_duration,
                max_center_offset = self.max_center_offset,
            )

            X[agg_count:agg_count+count, :, :] = shifted_waveforms
            
            
            if iteration is not None:

                with h5py.File(self.outdir / "signal.h5", "a") as g:
                    
                    g.create_dataset(f"{iteration:03d}_{name}", data=shifted_waveforms.numpy())
            
            agg_count += count
        
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
            plus=X[:,0,:],
            cross=X[:,1,:]
        )

        ht, target_snrs, rescale_factor = self.rescaler.forward(
            ht,
            target_snrs = self.snr_distro(total_counts)
        )

        if iteration is not None:
            with h5py.File(self.outdir / "signal.h5", "a") as g:
                
                g.create_dataset(f"{iteration:03d}_ht", data=ht.numpy())
        
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