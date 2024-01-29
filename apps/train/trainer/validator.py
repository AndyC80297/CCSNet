import torch
import logging

import numpy as np

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
        batch_size: int,
        sample_rate: int, 
        sqrtnum: int,
        sample_duration, 
        # validaton_type: [list, dict], # bg, glitch, bg+signal, glitch+signal
        # metric_method: [list, dict],  # Exculsion-socre, AUC-score
        # stream: bool = False # If true, stream in data by a sliding window on the active segments then save infos of i/o
    ):
        
        self.ifos = ifos
        self.num_ifos = len(ifos)
        self.batch_size = batch_size
        
        self.signals_dict = signals_dict
        self.ccsn_list = list(signals_dict.keys())
        self.num_ccsn_type = len(self.ccsn_list)
        self.siganl_sampled = self.num_ccsn_type * batch_size
        
        self.sqrtnum = sqrtnum
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.kernel_length = sample_rate * sample_duration
        # self.sample_data = int(batch_size * steps_per_epoch * sample_factor)

        dist_distro = Uniform(1, 1)
        
        distance = 0.1*dist_distro(self.sqrtnum ** 2)

        self.val_signal = {}
        for name in self.ccsn_list:
            
            time = self.signals_dict[name][0]
            quad_moment = self.signals_dict[name][1]            

            hp, hc, theta, phi = on_grid_pol_to_sim(
                quad_moment,
                sqrtnum
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
                
            self.val_signal[name] = hp_hc
    
    def recorder(
        iteration,
        model,
        noise_preds,
        distance,
        
    ):
        pass
    
    # @torch.no_grad
    def prediction(
        self,
        inputs,
        targets,
        # batch_size,
        model
    ):  
        with torch.no_grad():
            preds = []
            
            data_loader = forged_dataloader(
                inputs,
                targets,
                batch_size=self.sqrtnum
            )
            
            for i, (X, _) in data_loader:
                
                output = model(X)
                preds.append(output)
            
            return torch.cat(preds)
            
         
    def __call__(
        self,
        back_ground_display, # From BackGroundDisplay.__call__()
        batch_size, # 4
        steps_per_epoch, # 4
        # injector, # From Injector
        # noise_protion = [0.25, 0.25, 0.25, 0.25],
        injection_dict = None,
        model = None,
        max_distance = None,
        distance: dict= None,
        device="cuda"
        # copy: bool =  True
    ):
        
        model.to(device)
        
        noise_setting = {
            "noise": [1, 0, 0, 0],
            "l1_glitch": [0, 1, 0, 0],
            "h1_glitch": [0, 0, 1, 0],
            "simultaneous_glitch": [0, 0, 0, 1]
        }
        
        for mode, noise_protion in noise_setting.items():
            
            noise, targets = back_ground_display(
                batch_size = self.sqrtnum,
                steps_per_epoch = self.sqrtnum,
                glitch_dist = noise_protion,
                choice_mask = [0, 1, 2, 3],
                glitch_offset = 0.9,
                sample_factor = 1
                # target_value = 0
            )

            noise_prediction = self.prediction(
                noise, 
                targets, 
                # self.sqrtnum, 
                model
            )
            
            print(mode, noise_prediction.shape)
            logging.info(noise_prediction)
            # thereshold = torch.quantile(
            #     noise_prediction,
            #     0.75,
            # )

            # for name in self.ccsn_list:
            #     signal = noise + self.val_signal[name] / distance[name]
                
            


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
        