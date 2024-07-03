import h5py
import torch
import toml
import time
import logging

import numpy as np

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from ml4gw import gw
from ml4gw.transforms import SnrRescaler
from ml4gw.distributions import PowerLaw, Cosine, Uniform

from orchestrator import forged_dataloader
from ccsnet.utils import h5_thang
from ccsnet.waveform import CCSNe_Dataset
from ccsnet.waveform import get_hp_hc_from_q2ij, padding


def data_loader(
    signal,
    n_ifos=2, 
    sample_rate=4096,
    sample_duration=3,
    batch_size: int = 1024,
    shuffle=False,
    device="cpu",
    scaled_distance=None,
):

    dataset = CCSNe_Dataset(
        signal,
        scaled_distance,
        n_ifos=n_ifos, 
        sample_rate=sample_rate,
        sample_duration=sample_duration,
        device=device
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
        count: int,
        batch_size: int,
        sample_duration, 
        max_iteration: int, 
        output_dir: Path,
        signal_chopping:float=None, # Lable in second
        device: str ="cpu"
    ):
        
        self.ifos = ifos
        self.num_ifos = len(ifos)
        self.tensors, self.vertices = gw.get_ifo_geometry(*ifos)
        
        self.signals_dict = signals_dict
        self.chosen_signals = toml.load(chosen_signals)
        self.ccsn_list = list(signals_dict.keys())
        self.num_ccsn_type = len(self.ccsn_list)
        
        self.count = count #sqrtnum
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.max_iteration = max_iteration
        
        self.kernel_length = sample_rate * sample_duration
        self.output_dir = output_dir

        dec_distro = Cosine()
        psi_distro = Uniform(0, np.pi)
        phi_distro = Uniform(0, 2 * np.pi)

        ori_theta = np.random.uniform(0, np.pi, self.count)
        ori_phi = np.random.uniform(0, 2*np.pi, self.count)
        dec = dec_distro(self.count)
        psi = psi_distro(self.count)
        phi = phi_distro(self.count)

        self.snr_distro = Uniform(8, 12)
        distance = np.ones(self.count)

        self.val_signal = {}
        for name in self.ccsn_list:
            
            if signal_chopping is None:

                time = self.signals_dict[name][0]
                quad_moment = self.signals_dict[name][1] * 10 
                
            else:

                end = int((signal_chopping - self.signals_dict[name][0][0]) * sample_rate)
                time = self.signals_dict[name][0][:end]
                quad_moment = self.signals_dict[name][1][:end] * 10
            
            hp, hc = get_hp_hc_from_q2ij(
                quad_moment,
                theta=ori_theta,
                phi=ori_phi
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

            self.rescaler = SnrRescaler(
                num_channels=len(ifos), 
                sample_rate = sample_rate,
                waveform_duration = self.sample_duration,
                highpass = 32,
            )
            
            self.rescaler.fit(
                psds[0, :],
                psds[1, :],
                fftlength=fftlength,
                overlap=overlap,
                use_pre_cauculated_psd=True
            )
            
            # target_snrs, rescale_factor
            ht, _, _ = self.rescaler.forward( 
                ht,
                target_snrs = self.snr_distro(self.count)
            )
            
            self.val_signal[name] = ht
    
    def summarizer(
        self,
        iteration,
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

    
    def ccsne_loader(
            self,
            shuffle=False,
            device="cpu"
    ):

        num_family = len(self.ccsn_list)

        ccsne = torch.empty((self.count * num_family, self.num_ifos, self.kernel_length))

        for i, name in enumerate(self.ccsn_list):
            
            # This part can be modify to iteration dependent
            ccsne[i*self.count:(i+1)*self.count] = self.val_signal[name]


        signal_dataset = CCSNe_Dataset(
            signal=ccsne, 
            n_ifos=self.num_ifos, 
            sample_rate=self.sample_rate,
            sample_duration=self.sample_duration,
            device=device
        )
        
        return DataLoader(signal_dataset, batch_size=self.batch_size, shuffle=shuffle)

        

    def background_loader(
        self,
        inputs,
        shuffle=False,
        device="cpu"
    ):

        dataset = TensorDataset(
            inputs.to(device),
            # targets.view(-1, 1).to(torch.float).to(device)
        )
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    
    def __call__(
        self, 
        back_ground_display, 
        model, 
        whiten_model, 
        psds, 
        iteration=None, 
        signal_saving=None,
        device="cpu"
    ):

        noise_setting = {
            "No_Glitch": [1, 0, 0, 0],
            "L1_Glitch": [0, 1, 0, 0],
            "H1_Glitch": [0, 0, 1, 0],
            "Combined": [0, 0, 0, 1]
        }

        siganl_loader = self.ccsne_loader(device=device)

        for mode, noise_protion in noise_setting.items():
        
            steps_per_epoch = int(self.count / self.batch_size)

            if steps_per_epoch < 1:
                steps_per_epoch = 1

            noise, _ = back_ground_display(
                batch_size = self.batch_size,
                steps_per_epoch = steps_per_epoch,
                glitch_dist = noise_protion,
                choice_mask = [0, 1, 2, 3],
                glitch_offset = 0.9,
                sample_factor = 1,
                iteration=None,
                mode="Validate",
                target_value = 0,
            )
            
            noise_loader = self.background_loader(noise, device=device)

            with torch.no_grad():

                with h5py.File(self.output_dir/ "raw_data" / "history.h5", "a") as g:
                    h = g.create_group(f"Validation_itera{iteration:03d}_{mode}")

                    preds = []
                    inject_preds = []
                    for noise_data in noise_loader:

                        X = whiten_model(noise_data[0], psds)
                        noise_output = model(X)
                        
                        for siganl, _ in siganl_loader:

                            X = whiten_model(torch.add(noise_data[0], siganl[0]), psds)
                            signal_output = model(X)
                            
                            inject_preds.append(signal_output)
                        preds.append(noise_output)
                
                    h.create_dataset(f"noise", data=torch.cat(preds).detach().cpu().numpy().reshape(-1))
                    h.create_dataset(f"signal", data=torch.cat(inject_preds).detach().cpu().numpy().reshape(-1))
                

                logging.info(f"  {mode}:")
                logging.info(f"    Noise Average : {torch.cat(preds).mean():.02f}")
                logging.info(f"    Signal Average: {torch.cat(inject_preds).mean():.02f}")
                logging.info("")