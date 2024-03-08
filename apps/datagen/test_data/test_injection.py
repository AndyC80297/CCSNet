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

from test_orchestrator import Test_Injector

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

# test_bg = test_dir / ccsnet_args["test_background"] # Background + psd
test_data = test_dir / ccsnet_args["test_data"]
test_data.mkdir(parents=True, exist_ok=True)
# h5_reader(ccsnet_args["psd_files"]).h5_data()
# psds = torch.tensor(h5_thang(ccsnet_args["psd_files"]).h5_data()["psds"]).double()
psd_path = test_dir / ccsnet_args["test_psds"]
psds = torch.tensor(h5_thang(psd_path).h5_data(["psd"])["psd"]).double()
count = ccsnet_args["test_count"]
# Load BG

# Generate CCSN 
inited_injector = Test_Injector(
    ifos = ccsnet_args["ifos"],
    sample_rate = ccsnet_args["sample_rate"],
    sample_duration = ccsnet_args["sample_duration"],
    highpass = ccsnet_args["highpass"],
    psds = psds,
    fftlength = ccsnet_args["fftlength"],
    overlap = ccsnet_args["overlap"],
    count = ccsnet_args["count"],
    snr_distro = ccsnet_args["snr_distro"]
)


for key in signals_dict.keys():
    print(key)
    ht = inited_injector(
        time = signals_dict[key][0],
        quad_moment = signals_dict[key][1]
    )
    
    key_name = key.replace("/", "_")
    with h5py.File(test_data / f"{key_name}.h5", "w") as g:
        
        g.create_dataset("Signal", data=ht.numpy())
    # print(type(time))
    # print(type(quad_moment))
# Inject signal
# print(ccsnet_args["snr_distro"])
# Save data


# with h5py.File(test_file, "a") as g:
    
    # h = g.create_group()
    
    # g.attrs[""] = 