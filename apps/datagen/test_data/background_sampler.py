import toml
import h5py
import torch
import logging

import numpy as np

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from ml4gw import gw
from ml4gw.transforms import SnrRescaler
from ml4gw.distributions import PowerLaw, Cosine, Uniform

from ccsnet.utils import h5_thang, args_control
from ccsnet.waveform import get_hp_hc_from_q2ij, load_h5_as_dict, padding

from test_orchestrator import Test_BackGroundDisplay, random_ccsn_parameter

logging.basicConfig(level=logging.NOTSET)

parser = ArgumentParser()
parser.add_argument("-e", "--env", help="The env setting")
parser.add_argument("-s", "--seg", type=int, help="Testing segment") ## Segments/Runs_XX
args = parser.parse_args()

ccsnet_args = args_control(
    envs_file=args.env,
    test_segment=f"Seg{args.seg:02d}",
    saving=False
)

random_ccsn_parameter(
    count=ccsnet_args["test_count"],
    save_path=ccsnet_args["test_siganl_parameter"]
)

background_display = Test_BackGroundDisplay(
    ifos=ccsnet_args["ifos"],
    background_file=ccsnet_args["test_backgrounds"],
    sample_rate=ccsnet_args["sample_rate"],
    sample_duration=ccsnet_args["sample_duration"],
    test_seg=args.seg
)

background_list = {
    "No_Glitch": [1, 0, 0, 0],
    "H1_Glitch": [0, 0, 1, 0],
    "L1_Glitch": [0, 1, 0, 0],
    "Combined": [0, 0, 0, 1]
}

if not ccsnet_args["sampled_background"].is_file():

    with h5py.File(ccsnet_args["sampled_background"], "a") as g:

        for bg_key, bg_item in background_list.items():
            
            bg_data = background_display(
                glitch_dist=bg_item,
                test_count=ccsnet_args["test_count"],
            )

            g.create_dataset(name=bg_key, data=bg_data)