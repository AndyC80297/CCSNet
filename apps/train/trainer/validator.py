import h5py
import torch
import logging
import pickle 

import numpy as np

from pathlib import Path

from ml4gw import gw
from ml4gw.transforms import SnrRescaler
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
        psds,
        fftlength,
        overlap,
        sample_rate: int, 
        sqrtnum: int,
        sample_duration, 
        max_iteration: int, 
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
        self.max_iteration = max_iteration
        
        self.kernel_length = sample_rate * sample_duration
        self.output_dir = output_dir
        # self.sample_data = int(batch_size * steps_per_epoch * sample_factor)
        self.snr_distro = PowerLaw(12, 100, 3)
        dist_distro = Uniform(1, 1) # Just for scaling keep this to one preserve the distance to 10kpc
        
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
            )
            
            ### This step would have to be fixed and it be sampled outside of the loop
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
                plus=torch.Tensor(hp_hc[:,0,:]),
                cross=torch.Tensor(hp_hc[:,1,:])
            )

            # rescaler = SnrRescaler(
            #     num_channels=len(ifos), 
            #     sample_rate = sample_rate,
            #     waveform_duration = self.sample_duration,
            #     highpass = 32,
            # )
            
            # rescaler.fit(
            #     psds[0, :],
            #     psds[1, :],
            #     fftlength=fftlength,
            #     overlap=overlap,
            #     use_pre_cauculated_psd=True
            # )
            
            # ht, target_snrs, rescale_factor = rescaler.forward(
            #     ht,
            #     target_snrs = self.snr_distro(sqrtnum ** 4)
            # )
            
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
        iteration,
        tpr_dict,
        max_distance,
        output_dir,
        noise_mode = "noise"
        
    ):
        

            
        for name, distance in max_distance.items():
            
            if tpr_dict[name][1] >= 0.5:
            
                max_distance[name] += 5
            
        if (iteration + 1) == self.max_iteration:
        
            with open(output_dir / "Max_Distance.pkl", "wb") as f:
                pickle.dump(max_distance, f)
                
            
        
        return max_distance
    
    # @torch.no_grad
    def prediction(
        self,
        inputs,
        targets,
        # batch_size,
        model,
        criterion,
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
                batch_size=self.sqrtnum**4, 
                shuffle=False
            )
            # print(__file__, "Type of the psd", type(psds.to("cuda")))
            for x, y in data_loader:
                
                x = whiten_model(
                    x, 
                    psds
                )
                
                output = model(x)
                loss = criterion(output, y)
                preds.append(output)
                
            # print(loss.item())
            return torch.cat(preds)
            
        

    def __call__(
        self, 
        loss,
        back_ground_display, 
        model, 
        criterion,
        whiten_model, 
        psds, 
        iteration, 
        max_distance = None, 
        # output_dir=None, 
        device="cuda"
    ):
        
        
        with h5py.File(self.output_dir/ "loss.h5", "a") as g:
            
            h = g.create_group(f"Itera{iteration:03d}/")
            
            h.create_dataset(f"loss", data=np.array([loss]))
            
        print(max_distance)
        noise_setting = {
            "noise": [1, 0, 0, 0],
            "l1_glitch": [0, 1, 0, 0],
            "h1_glitch": [0, 0, 1, 0],
            "simultaneous_glitch": [0, 0, 0, 1]
        }
        print()
        for mode, noise_protion in noise_setting.items():
            
            noise, targets = back_ground_display(
                batch_size = self.sqrtnum ** 4,
                steps_per_epoch = 1,
                glitch_dist = noise_protion,
                choice_mask = [0, 1, 2, 3],
                glitch_offset = 0.9,
                sample_factor = 1,
                iteration=iteration,
                mode=f"validation_{mode}",
                target_value = 0
            )
            
            print(f"Mode: {mode.upper()}:")
            # print("It's just some noise", noise)
            # print()
            noise_prediction = self.prediction(
                noise, 
                targets,
                model,
                criterion,
                whiten_model,
                psds,
            )
            
            with h5py.File(self.output_dir/"history.h5", "a") as g:
                
                h = g.create_group(f"Itera{iteration:03d}/{mode}")
                
                h.create_dataset(f"{mode}", data=noise_prediction.detach().cpu().numpy().reshape(-1))
                
            thereshold = torch.quantile(
                noise_prediction,
                torch.tensor([0.50, 0.95, 0.99]).to(device)
            )
            
            tpr_dict = {}
            for name in self.ccsn_list:
                # print(name)
                signal = noise + self.val_signal[name] / max_distance[name]
                
                # if mode == "noise":
                #     with h5py.File(self.output_dir/"val_signal.h5", "a") as g:
                
                #         h = g.create_group(f"Itera{iteration:03d}/{name}")
                
                #         h.create_dataset(f"Signal", data=self.val_signal[name].numpy())
                
                injection_prediction = self.prediction(
                    signal, 
                    torch.ones_like(targets),
                    model,
                    criterion,
                    whiten_model,
                    psds,
                )
                
                tprs = (injection_prediction >= thereshold).sum(0)/len(injection_prediction)
                print("    ", name, "TPR:", tprs.detach().cpu().numpy())
                tpr_dict[name] = tprs.detach().cpu().numpy()
                with h5py.File(self.output_dir/"history.h5", "a") as g:
                    
                    h = g[f"Itera{iteration:03d}/{mode}"]
                    
                    h.create_dataset(f"{name}", data=injection_prediction.detach().cpu().numpy().reshape(-1))
            print()    
                
                
        return self.summarizer(iteration, tpr_dict, max_distance, self.output_dir)
    
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
        