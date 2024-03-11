import toml
import h5py
import torch

import numpy as np

from tqdm import tqdm
from pathlib import Path

from ml4gw import gw
from ml4gw.transforms import SnrRescaler
from ml4gw.distributions import PowerLaw, Cosine, Uniform

from ccsnet.utils import h5_thang
from ccsnet.waveform import get_hp_hc_from_q2ij, load_h5_as_dict, padding

from test_orchestrator import Test_BackGroundDisplay, Test_Injector

ARGUMENTS_FILE = "/home/hongyin.chen/anti_gravity/CCSNet/apps/train/trainer/arguments.toml"

ccsnet_args = toml.load(ARGUMENTS_FILE)

data_dir = Path(ccsnet_args["data_dir"])
test_dir = Path(ccsnet_args["test_dir"])

test_dir.mkdir(parents=True, exist_ok=True)
signals_dir = data_dir / ccsnet_args["signals_dir"]

signals_dict = load_h5_as_dict(
    ccsnet_args["chosen_signals"],
    signals_dir
)

test_bg = test_dir / ccsnet_args["test_background"] # Background
test_data_dir = test_dir / ccsnet_args["test_data"]
glitch_info = test_dir / ccsnet_args["test_glitch_info"]
test_data_dir.mkdir(parents=True, exist_ok=True)


psd_path = test_dir / ccsnet_args["test_psds"]
psds = torch.tensor(h5_thang(psd_path).h5_data(["psd"])["psd"]).double()

count = ccsnet_args["test_count"]

background = torch.tensor(h5_thang(test_bg).h5_data()["segments00/strain"])
attrs = h5_thang(test_bg).h5_attrs()

test_start = attrs["segments00/start"]
test_end = attrs["segments00/end"]

background_display = Test_BackGroundDisplay(
    ifos = ccsnet_args["ifos"],
    background = background,
    background_dur = test_end - test_start,
    start_time = test_start,
    glitch_info = glitch_info,
    sample_rate = ccsnet_args["sample_rate"],
    sample_duration = ccsnet_args["sample_duration"]
)

inited_injector = Test_Injector(
    ifos = ccsnet_args["ifos"],
    sample_rate = ccsnet_args["sample_rate"],
    sample_duration = ccsnet_args["sample_duration"],
    highpass = ccsnet_args["highpass"],
    psds = psds,
    fftlength = ccsnet_args["fftlength"],
    overlap = ccsnet_args["overlap"],
    count = ccsnet_args["count"],
    snr_distro = ccsnet_args["snr_distro"],
    buffer_duration = 4,
    off_set=0.15,
    save_dir = test_data_dir
)




# # Load BG
background_list = {
    "No_Glitch": [1, 0, 0, 0],
    "H1_Glitch": [0, 0, 1, 0],
    "L1_Glitch": [0, 1, 0, 0],
    "Combined": [0, 0, 0, 1]
}

for bg_key, bg_item in background_list.items():
    
    bg_data, _ = background_display(
        glitch_dist=bg_item,
        batch_size=ccsnet_args["count"],
        steps_per_epoch=1,
        sample_factor=1
    )


    with h5py.File(test_data_dir / f"{bg_key}.h5", "w") as g:
        
        g.create_dataset("Signal", data=bg_data.numpy())
        # g.create_dataset("distance", data=1/rescale_factor.numpy())
        # g.attrs["SNR"] = snr
        
        # Generate CCSN 

    for key in signals_dict.keys():
        
        data_gen = inited_injector(
            time = signals_dict[key][0],
            quad_moment = signals_dict[key][1]
        )
        
        key_name = key.replace("/", "_")
        
        for snr, ht, rescale_factor in data_gen:
            
            with h5py.File(test_data_dir / f"{bg_key}_{key_name}_{snr:02d}.h5", "w") as g:
                
                ht += bg_data
                g.create_dataset("Signal", data=ht.numpy())
                g.create_dataset("distance", data=1/rescale_factor.numpy())
                g.attrs["SNR"] = snr

# # Inject signal

# # Save data


# # with h5py.File(test_file, "a") as g:
    
#     # h = g.create_group()
    
#     # g.attrs[""] = 