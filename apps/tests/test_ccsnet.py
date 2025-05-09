import sys
import h5py
import time
import torch
import logging

import numpy as np

from tqdm import tqdm
from pathlib import Path
from tqdm.contrib import tzip
from argparse import ArgumentParser
from torch.utils.data import TensorDataset, DataLoader

from ccsnet.arch import WaveNet
from ccsnet.utils import h5_thang, args_control
from ccsnet.waveform import load_h5_as_dict, Waveform_Projector

from model_streamer import Streamer, test_data_loader, strain_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser()
parser.add_argument("-e", "--env", help="The env setting")
parser.add_argument("-a", "--agrs_file", help="The argument setting")
parser.add_argument("-s", "--seg", type=int, help="Testing segment") ## Segments/Runs_XX
parser.add_argument("-r", "--run", help="Running count")
args = parser.parse_args()

ccsnet_args = args_control(
    envs_file=args.env,
    agrs_file=args.agrs_file,
    test_segment=f"Seg{args.seg:03d}/{args.run}",
    saving=False
)


logging.info(f"Running Seg{args.seg:03d}")


num_ifos = len(ccsnet_args["ifos"])

if ccsnet_args.get("coh_mode") == "remove": 
    
    num_ifos = 1 

ccsnet_streamer = Streamer(
    num_ifos=num_ifos,
    sample_rate=ccsnet_args["sample_rate"],
    architecture=WaveNet,
    model_weights=ccsnet_args["test_model"],
    fftlength=ccsnet_args["fftlength"],
    highpass=ccsnet_args["highpass"],
    test_psd_seg=ccsnet_args["test_psd_seg"],
    coh_ifo=ccsnet_args.get("coh_ifo"),
    coh_mode=ccsnet_args.get("coh_mode"),
    map_device="cpu",
    device=device
)

inited_injector = Waveform_Projector(
    ifos=ccsnet_args["ifos"],
    sample_rate=ccsnet_args["sample_rate"],
    background_file=ccsnet_args["backgrounds"],
    seg=f"segments{args.seg:03d}",
    highpass=ccsnet_args["highpass"],
    fftlength=ccsnet_args["fftlength"],
    overlap=ccsnet_args["overlap"],
    sample_duration=ccsnet_args["sample_duration"],
    test_psd_seg=ccsnet_args["test_psd_seg"],
    buffer_duration=4,
    time_shift=0,
    off_set=ccsnet_args["off_set"]
)

siganl_parameter = h5_thang(ccsnet_args["test_siganl_parameter"]).h5_data()

signals_dict = load_h5_as_dict(
    ccsnet_args["test_signals"],
    ccsnet_args["signals_dir"]
)

sampled_background = h5_thang(ccsnet_args["sampled_background"]).h5_data()

count = ccsnet_args["test_count"]

start = time.time()
with h5py.File(ccsnet_args["test_result_dir"] / "background_result.h5", "w") as g:

    with torch.no_grad():
        logging.info("")
        logging.info(f"= - = - = - = - =")
        logging.info(f"Coherency key activate on {ccsnet_args.get('coh_ifo')}")
        for mode, raw_bg in sampled_background.items():
            if mode in ["H1_Glitch", "L1_Glitch"]:
                continue
            noise_loader = test_data_loader(
                torch.Tensor(raw_bg[:count, :,:]),
                batch_size=ccsnet_args["test_batch_size"],
                device=device
            )

            preds = []
            
            for noise_data, _ in noise_loader:
                    
                pred = ccsnet_streamer.stream(
                    noise_data, 
                    psd=None
                )
                
                preds.append(pred.cpu().detach().numpy())

            prediction = np.concatenate(preds).reshape([-1])
            logging.info(f"{mode}: average performance {prediction.mean():.02f}")
            h = g.create_dataset(name=f"{mode}", data=prediction)
        logging.info(f"= - = - = - = - =")
        logging.info("")

with h5py.File(ccsnet_args["test_result_dir"] / "injection_result.h5", "w") as g:

    for key in signals_dict.keys():
        logging.info(f"Running {key} analysis")
        g1 = g.create_group(key)
        
        scaled_ht, distance = inited_injector(
            time = signals_dict[key][0],
            quad_moment = torch.Tensor(signals_dict[key][1] * 10),
            ori_theta=torch.tensor(siganl_parameter["ori_theta"][:count]),
            ori_phi=torch.tensor(siganl_parameter["ori_phi"][:count]),
            dec=torch.tensor(siganl_parameter["dec"][:count]),
            psi=torch.tensor(siganl_parameter["psi"][:count]),
            phi=torch.tensor(siganl_parameter["phi"][:count]),
        )

        g1.create_dataset(name="SNR_4_Distance", data=distance.numpy())
        test_loop_start = time.time()
        with torch.no_grad():
            
            ccsn_loader = test_data_loader(
                signal=scaled_ht,
                scaled_distance=distance,
                batch_size=ccsnet_args["test_batch_size"],
                device=device
            )

            for mode, raw_bg in tqdm(sampled_background.items()):
                if mode in ["H1_Glitch", "L1_Glitch"]:
                    continue
                logging.info(f"Mode: {mode}")
                g2 = g1.create_group(mode)

                noise_loader = test_data_loader(
                    torch.Tensor(raw_bg[:count, :,:]),
                    batch_size=ccsnet_args["test_batch_size"],
                    device=device
                )
                
                for snr in ccsnet_args["snr_distro"]:

                    preds = []
                    for background, siganl in zip(noise_loader, ccsn_loader):
                        
                        factor = (snr/4)
                        X = background[0] + siganl[0] * factor
        
                        pred = ccsnet_streamer.stream(
                            X, 
                            psd=None
                        )
                        
                        preds.append(pred.cpu().detach().numpy())
                    
                    prediction = np.concatenate(preds).reshape([-1])
                    logging.info(f"    SNR: {snr} average performance {prediction.mean():.02f}")
                    g2.create_dataset(name=f"SNR_{snr:02d}", data=prediction)

                
                max_dis = distance.mean()
                prob_dis = np.geomspace(max_dis/50, max_dis*2, 20)
                logging.info("")
                for dis_i, dis in enumerate(prob_dis):
                    
                    preds = []
                    count_down = 0
                    for background, siganl in zip(noise_loader, ccsn_loader):
                        
                        scale_back_ht = torch.einsum('li, lik->lik', siganl[1], siganl[0]) / dis
                        X = background[0] + scale_back_ht
                            
                        pred = ccsnet_streamer.stream(
                            X, 
                            psd=None
                        )
                        
                        preds.append(pred.cpu().detach().numpy())
                    
                    prediction = np.concatenate(preds).reshape([-1])
                    
                    logging.info(f"    Dis: {dis:.03f} average performance {prediction.mean():.02f}")
                    g2.create_dataset(name=f"Distance_{dis_i:02d}", data=prediction)
                g2.create_dataset(name=f"Distances", data=prob_dis)
        
logging.info(f"Time spent on test is {int(time.time() - start)/60:.02f} min.")