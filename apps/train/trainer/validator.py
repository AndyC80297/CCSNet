import h5py
import torch
import toml
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
        chosen_signals,
        psds,
        fftlength,
        overlap,
        sample_rate: int, 
        sqrtnum: int,
        sample_duration, 
        max_iteration: int, 
        output_dir: Path,
        device: str ="cuda"
    ):
        
        self.ifos = ifos
        self.num_ifos = len(ifos)
        self.tensors, self.vertices = gw.get_ifo_geometry(*ifos)
        
        self.signals_dict = signals_dict
        self.chosen_signals = toml.load(chosen_signals)
        self.ccsn_list = list(signals_dict.keys())
        self.num_ccsn_type = len(self.ccsn_list)
        
        self.sqrtnum = sqrtnum
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.max_iteration = max_iteration
        
        self.kernel_length = sample_rate * sample_duration
        self.output_dir = output_dir

        dec_distro = Cosine()
        psi_distro = Uniform(0, np.pi)
        phi_distro = Uniform(0, 2 * np.pi)

        dec = dec_distro(sqrtnum ** 4)
        psi = psi_distro(sqrtnum ** 4)
        phi = phi_distro(sqrtnum ** 4)

        self.snr_distro = Uniform(8, 12)
        distance = np.ones(self.sqrtnum ** 4)

        self.val_signal = {}
        for name in self.ccsn_list:
            
            time = self.signals_dict[name][0]
            quad_moment = self.signals_dict[name][1]            

            hp, hc, ori_theta, ori_phi = on_grid_pol_to_sim(
                quad_moment * 0.1,
                sqrtnum ** 2
            )
            
            hp_hc = padding(
                time,
                hp,
                hc,
                distance,
                sample_kernel = self.sample_duration, 
            )
            
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

            rescaler = SnrRescaler(
                num_channels=len(ifos), 
                sample_rate = sample_rate,
                waveform_duration = self.sample_duration,
                highpass = 32,
            )
            
            rescaler.fit(
                psds[0, :],
                psds[1, :],
                fftlength=fftlength,
                overlap=overlap,
                use_pre_cauculated_psd=True
            )
            
            # target_snrs, rescale_factor
            ht, _, _ = rescaler.forward( 
                ht,
                target_snrs = self.snr_distro(sqrtnum ** 4)
            )
            
            self.val_signal[name] = ht

    
    def summarizer(
        self,
        iteration,
        max_distance = None,
        factor = [0.50, 0.95, 0.99],
        noise_mode = "noise"
    ):
        
        score = torch.empty([3, len(self.ccsn_list), 4])

        modes = [
            "noise",
            "h1_glitch",
            "l1_glitch",
            "simultaneous_glitch"
        ]

        history = h5_thang(self.output_dir / "raw_data" / "history.h5")

        for i, mode in enumerate(modes):
            
            trace_tag = f"Itera{iteration:03d}/{mode}/{mode}"
            noise_pred = torch.tensor(history.h5_data([trace_tag])[trace_tag])

            thereshold = torch.quantile(
                noise_pred,
                torch.tensor(factor)
            )

            sig_count = 0
            for family in self.chosen_signals.keys():
                for name in self.chosen_signals[family]:

                    full_name = f"{family}/{name}"

                    trace_tag = f"Itera{iteration:03d}/{mode}/{full_name}"
                    
                    sig_out = torch.tensor(history.h5_data([trace_tag])[trace_tag])
                
                    tprs = (sig_out.view(self.sqrtnum ** 4, 1) >= thereshold).sum(0)/len(sig_out)

                    score[:, sig_count, i] = tprs
                    sig_count += 1

        with h5py.File(self.output_dir / "val_performance.h5", "a") as g:

            g.create_dataset(f"Itera{iteration:03d}_score", data=score.numpy())

        noise_score = score[1].mean(axis=0)

        for i, mode in enumerate(modes):

            logging.info(f"  Mode: {mode.upper():15s}  {noise_score[i]:.04f}")

        wave_score = score[1].mean(axis=1)
        wave_count = 0
        for family in self.chosen_signals.keys():
            logging.info(f"  {family}")
            for name in self.chosen_signals[family]:

                full_name = f"{family}/{name}"

                logging.info(f"    {name:20s} SNR:8-12   TPR:{wave_score[wave_count]:.04f}")

                wave_count += 1
        
        if max_distance is not None:
            return max_distance
        

    

    def prediction(
        self,
        inputs,
        targets,
        model,
        whiten_model,
        psds
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

            for x, y in data_loader:
                
                x = whiten_model(
                    x, 
                    psds
                )
                
                output = model(x)
                preds.append(output)
            
            return x, torch.cat(preds)
            


    def __call__(
        self, 
        back_ground_display, 
        model, 
        criterion,
        whiten_model, 
        psds, 
        iteration, 
        max_distance=None,
        signal_saving=None,
        device="cpu"
    ):


        if max_distance is not None:

            with h5py.File(self.output_dir/ "raw_data" /"max_distance.h5", "a") as g:
                
                h = g.create_group(f"Itera{iteration:03d}")
                for name in self.ccsn_list:

                    h.create_dataset(f"{name}", data=np.array([max_distance[name]]))

        noise_setting = {
            "noise": [1, 0, 0, 0],
            "l1_glitch": [0, 1, 0, 0],
            "h1_glitch": [0, 0, 1, 0],
            "simultaneous_glitch": [0, 0, 0, 1]
        }

        for mode, noise_protion in noise_setting.items():
            
            noise, targets = back_ground_display(
                batch_size = self.sqrtnum ** 4,
                steps_per_epoch = 1,
                glitch_dist = noise_protion,
                choice_mask = [0, 1, 2, 3],
                glitch_offset = 0.9,
                sample_factor = 1,
                # iteration=iteration,
                mode="Validate",
                target_value = 0,
                # noise_mode=mode

            )
            
            val_data, noise_prediction = self.prediction(
                noise, 
                targets,
                model,
                whiten_model,
                psds,
            )

            with h5py.File(self.output_dir/ "raw_data" / "history.h5", "a") as g:
                
                h = g.create_group(f"Itera{iteration:03d}/{mode}")
                
                h.create_dataset(f"{mode}", data=noise_prediction.detach().cpu().numpy().reshape(-1))
                # h.create_dataset(f"{mode}_siganl", data=val_data.detach().cpu().numpy())
                

            for name in self.ccsn_list:
                
                if max_distance is not None:
                    
                    signal = noise + self.val_signal[name] / max_distance[name]
                else:
                    
                    signal = noise + self.val_signal[name]

                val_data, injection_prediction = self.prediction(
                    signal, 
                    torch.ones_like(targets),
                    model,
                    whiten_model,
                    psds,
                )


                with h5py.File(self.output_dir/ "raw_data" / "history.h5", "a") as g:
                    
                    h = g[f"Itera{iteration:03d}/{mode}"]
                    
                    h.create_dataset(f"{name}", data=injection_prediction.detach().cpu().numpy().reshape(-1))

                if signal_saving is not None:

                    if mode == "noise":
                        with h5py.File(self.output_dir / "raw_data/val_signal.h5", "a") as g:
                    
                            h = g.create_group(f"Itera{iteration:03d}/{name}")
                             

        if  max_distance is not None:
            return self.summarizer(iteration)
        self.summarizer(iteration)

        