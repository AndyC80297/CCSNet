import h5py
import torch
import logging

import numpy as np

from pathlib import Path

from ml4gw import gw
from ml4gw.distributions import PowerLaw, Cosine, Uniform

from orchestrator import forged_dataloader
from ccsnet.utils import h5_thang
from ccsnet.waveform import on_grid_pol_to_sim, padding


class Validator:
    
    def __init__(
        self,
        ifos,
        signals_dict,
        # validation_fraction,
        # injection_control, # Distance, SNR, time_shift ### We might need another function here (ML4GW)
        # batch_size: int,
        sample_rate: int, 
        sqrtnum: int,
        sample_duration, 
        output_dir: Path,
        device: str ="cuda"
        # validaton_type: [list, dict], # bg, glitch, bg+signal, glitch+signal
        # metric_method: [list, dict],  # Exculsion-socre, AUC-score
        # stream: bool = False # If true, stream in data by a sliding window on the active segments then save infos of i/o
    ):
        
        self.ifos = ifos
        self.num_ifos = len(ifos)
        self.tensors, self.vertices = gw.get_ifo_geometry(*ifos)
        # self.batch_size = batch_size
        
        self.signals_dict = signals_dict
        self.ccsn_list = list(signals_dict.keys())
        self.num_ccsn_type = len(self.ccsn_list)
        # self.siganl_sampled = self.num_ccsn_type * batch_size
        
        self.sqrtnum = sqrtnum
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.kernel_length = sample_rate * sample_duration
        self.output_dir = output_dir
        # self.sample_data = int(batch_size * steps_per_epoch * sample_factor)

        dist_distro = Uniform(1, 1)
        
        distance = 0.1*dist_distro(self.sqrtnum ** 4)

        self.val_signal = {}
        for name in self.ccsn_list:
            
            time = self.signals_dict[name][0]
            quad_moment = self.signals_dict[name][1]            

            hp, hc, theta, phi = on_grid_pol_to_sim(
                quad_moment,
                sqrtnum ** 2
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
            
            hp_hc = torch.Tensor(hp_hc)
            
            dec_distro = Cosine()
            psi_distro = Uniform(0, np.pi)
            phi_distro = Uniform(0, 2 * np.pi)
            
            dec = dec_distro(sqrtnum ** 4)
            psi = psi_distro(sqrtnum ** 4)
            phi = phi_distro(sqrtnum ** 4)
            
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


            self.val_signal[name] = ht
    
    def recorder(
        self,
        iteration,
        model,
        mode,
        name,
        noise_preds,
        distance,
        output_dir
    ):
        
        pass
    
    def summarizer(
        self,
        max_distance,
        output_dir
        
    ):
        
        for name, distance in max_distance.items():
            
            # if tprs[1] >= tprs[1]:
            
            max_distance[name] += 1
        
        
        return max_distance
    # @torch.no_grad
    def prediction(
        self,
        inputs,
        targets,
        # batch_size,
        model,
        whiten_model,
        psds,
    ):  
        with torch.no_grad():
            preds = []
            
            dataset = torch.utils.data.TensorDataset(
                inputs.to("cuda"),
                targets.view(-1, 1).to(torch.float).to("cuda")
            )
            
            data_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.sqrtnum, 
                shuffle=False
            )
            
            for X, _ in data_loader:
                
                X = whiten_model(
                    X, 
                    psds
                )
                
                output = model(X)
                preds.append(output)
            
            return torch.cat(preds)
            
         
    def __call__(
        self, 
        back_ground_display, # From BackGroundDisplay.__call__()
        model, 
        whiten_model, 
        psds, 
        iteration, 
        max_distance = None, 
        # output_dir=None, 
        device="cuda"
    ):
        
        print(max_distance)
        noise_setting = {
            "noise": [1, 0, 0, 0],
            "l1_glitch": [0, 1, 0, 0],
            "h1_glitch": [0, 0, 1, 0],
            "simultaneous_glitch": [0, 0, 0, 1]
        }
        
        for mode, noise_protion in noise_setting.items():
            
            noise, targets = back_ground_display(
                batch_size = self.sqrtnum ** 2,
                steps_per_epoch = self.sqrtnum ** 2,
                glitch_dist = noise_protion,
                choice_mask = [0, 1, 2, 3],
                glitch_offset = 0.9,
                sample_factor = 1
                # target_value = 0
            )

            noise_prediction = self.prediction(
                noise, 
                targets,
                model,
                whiten_model,
                psds,
            )
            
            with h5py.File(self.output_dir/"history.h5", "a") as g:
                
                h = g.create_group(f"Itera{iteration:03d}/{mode}")
                
                h.create_dataset(f"{mode}", data=noise_prediction.detach().cpu().numpy().reshape(-1))

            thereshold = torch.quantile(
                noise_prediction,
                torch.tensor([0.1, 0.75, 0.99]).to(device)
            )

            for name in self.ccsn_list:
                # print(name)
                signal = noise + self.val_signal[name] / max_distance[name]
                
                injection_prediction = self.prediction(
                    signal, 
                    torch.ones_like(targets),
                    model,
                    whiten_model,
                    psds,
                )

                # tprs = (injection_prediction > thereshold).sum(0)/len(injection_prediction)
                # print(name, "TPR:", tprs.detach().cpu().numpy())
                
                with h5py.File(self.output_dir/"history.h5", "a") as g:
                    
                    h = g[f"Itera{iteration:03d}/{mode}"]
                    
                    h.create_dataset(f"{name}", data=injection_prediction.detach().cpu().numpy().reshape(-1))
                
                
                
        return self.summarizer(max_distance, self.output_dir)
    
        # return early_stopping, max_distance
# Read CCSN h5 files
# Label each CCSN them by name

# Sample BG, Glitch, CCSN
# Injection rescalling
# Whiten, Cropping

####################
### Model Stream ###
####################

# This function should interact with validation scheme
### The validation scheme should includes 
# (1). Recall at var(??)% of glitch exclusion
# (2). 
# def forward ...
#   return valdation result
        