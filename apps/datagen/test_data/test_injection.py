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

ARGUMENTS_FILE = "/home/hongyin.chen/anti_gravity/CCSNet/apps/train/trainer/arguments.toml"

ccsnet_args = toml.load(ARGUMENTS_FILE)

# test_dir = Path(ccsnet_args["test_dir"])
# test_dir.mkdir(parents=True, exist_ok=True)

# test_bg = test_dir / ccsnet_args["test_background"] # Background + psd
# test_file = test_dir / ccsnet_args["test_data"]
# h5_reader(ccsnet_args["psd_files"]).h5_data()
# psds = torch.tensor(h5_thang(ccsnet_args["psd_files"]).h5_data()["psds"]).double()
psds = torch.tensor(h5_thang(ccsnet_args["psd_files"]).h5_data(["psd"])["psd"]).double()

# Load BG

# Generate CCSN 

ifos = ccsnet_args["ifos"]
sample_rate = ccsnet_args["sample_rate"]
sample_duration = ccsnet_args["sample_duration"]
highpass = ccsnet_args["highpass"]

fftlength = ccsnet_args["fftlength"]
overlap = ccsnet_args["overlap"]

rescaler = SnrRescaler(
    num_channels=len(ifos), 
    sample_rate = sample_rate,
    waveform_duration = sample_duration,
    highpass = 32,
)

rescaler.fit(
    psds[0, :],
    psds[1, :],
    fftlength=fftlength,
    overlap=overlap,
    use_pre_cauculated_psd=True
)
# Inject signal

# Save data


# with h5py.File(test_file, "a") as g:
    
    # h = g.create_group()
    
    # g.attrs[""] = 