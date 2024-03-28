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

from model_streamer import Streamer, test_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.NOTSET)

parser = ArgumentParser()
parser.add_argument("-e", "--env", help="The env setting")
parser.add_argument("-s", "--seg", type=int, help="Testing segment") ## Segments/Runs_XX
parser.add_argument("-r", "--run", help="Running count")
args = parser.parse_args()

ccsnet_args = args_control(
    envs_file=args.env,
    test_segment=f"Seg{args.seg:02d}/{args.run}",
    saving=False
)

ccsnet_streamer = Streamer(
    num_ifos=len(ccsnet_args["ifos"]),
    sample_rate=ccsnet_args["sample_rate"],
    architecture=WaveNet,
    model_weights=ccsnet_args["test_model"],
    fftlength=ccsnet_args["fftlength"],
    highpass=ccsnet_args["highpass"],
    test_seg=args.seg,
    background_file=ccsnet_args["test_backgrounds"], # ccsnet_args["test_backgrounds"]
    map_device="cpu",
    device=device
)

inited_injector = Waveform_Projector(
    ifos=ccsnet_args["ifos"],
    sample_rate=ccsnet_args["sample_rate"],
    background_file=ccsnet_args["test_backgrounds"],
    seg=f"segments{args.seg - 1 :02d}",
    highpass=ccsnet_args["highpass"],
    fftlength=ccsnet_args["fftlength"],
    overlap=ccsnet_args["overlap"],
    sample_duration=ccsnet_args["sample_duration"],
    buffer_duration=3,
    time_shift=0,
    off_set=0
)

siganl_parameter = h5_thang(ccsnet_args["test_siganl_parameter"]).h5_data()

signals_dict = load_h5_as_dict(
    ccsnet_args["test_siganls"],
    ccsnet_args["signals_dir"]
)

sampled_background = h5_thang(ccsnet_args["sampled_background"]).h5_data()

count = ccsnet_args["test_count"]

start = time.time()
with h5py.File(ccsnet_args["test_result_dir"] / "background_result.h5", "w") as g:

    with torch.no_grad():

        logging.info(f"= - = - = - = - =")

        for mode, raw_bg in sampled_background.items():

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

with h5py.File(ccsnet_args["test_result_dir"] / "injection_result.h5", "w") as g:

    for key in tqdm(signals_dict.keys()):

        g1 = g.create_group(key)
        scaled_ht, distance = inited_injector(
            time = signals_dict[key][0],
            quad_moment = signals_dict[key][1],
            ori_theta=siganl_parameter["ori_theta"][:count],
            ori_phi=siganl_parameter["ori_phi"][:count],
            dec=siganl_parameter["dec"][:count],
            psi=siganl_parameter["psi"][:count],
            phi=siganl_parameter["phi"][:count],
        )
        g1.create_dataset(name="SNR_4_Distance", data=distance.numpy())
        with torch.no_grad():
            
            ccsn_loader = test_data_loader(
                scaled_ht,
                batch_size=ccsnet_args["test_batch_size"],
                device=device
            )

            for mode, raw_bg in sampled_background.items():
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

                    g2.create_dataset(name=f"SNR_{snr:02d}", data=prediction)

logging.info(f"Time spent on test is {int(time.time() - start)/60:.02f} min.")