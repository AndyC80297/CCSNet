import toml
import torch

from pathlib import Path
from argparse import ArgumentParser

from ccsnet.omicron import glitch_merger, psd_estimiater
from ccsnet.utils import h5_thang, args_control

parser = ArgumentParser()
parser.add_argument("-e", "--env", help="The env setting")
args = parser.parse_args()

ccsnet_args = args_control(
    args.env,
)

glitch_merger(
    ifos=ccsnet_args["ifos"],
    omicron_path=Path("/home/hongyin.chen/anti_gravity/anomaly_detection/CCSNet_test_01"),
    channels=ccsnet_args["channels"],
    output_file=ccsnet_args["test_glitch_info"]
)


def main(
    # Segments & General
    ifos,
    start,
    end,
    sample_rate,
    frame_type,
    state_flag,
    channels,

    # Gltich
    omicron_path,

    # Get Strain

    # PSD
    strains,
    kernel_width,
    fftlength,
    overlap,

    # Saving
    output_file: Path,

):


    # Read glitch file from omicron triggers
    
    num_ifo = len(ifos)

    query_flag = []
    for i, ifo in enumerate(ifos):
        query_flag.append(f"{ifo}:{state_flag[i]}")

    flags = DataQualityDict.query_dqsegdb(
        query_flag,
        start,
        end
    )

    for contents in flags.intersection().active.to_table():
        
        idx = contents["index"]
        start = contents["start"]
        end = contents["end"]