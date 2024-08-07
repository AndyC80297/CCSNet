import h5py
import toml
import torch

import numpy as np
import logging

from pathlib import Path
from gwdatafind import find_urls
from argparse import ArgumentParser
from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityDict

from ccsnet.utils import h5_thang, args_control
from ccsnet.omicron import glitch_merger, psd_estimiater

from run_omicron import get_conincident_segs

parser = ArgumentParser()
parser.add_argument("-e", "--env", help="The env setting")
args = parser.parse_args()

ccsnet_args = args_control(
    args.env,
)

def get_background(
    seg:tuple,
    ifos:list,
    frame_type:list,
    channels:list,
    sample_rate:int,
    verbose:bool=True
): 
    
    num_ifo = len(ifos)
    
    strain = []
    print(f"Collecting strain data from {seg[0]} to {seg[1]} at {channels}")
    for num in range(num_ifo):
    
        files = find_urls(
            site=f"{ifos[num][0]}",
            frametype=f"{ifos[num]}_{frame_type[num]}",
            gpsstart=seg[0],
            gpsend=seg[1],
            urltype="file",
        )
        
        strain.append(
            TimeSeries.read(
                files,
                f"{ifos[num]}:{channels[num]}",
                start=seg[0],
                end=seg[1],
                nproc=8,
                verbose=verbose
            ).resample(sample_rate).value
        )
    
    strain = np.stack(strain)

    return strain


def main(
    seg,
    seg_num,
    backgrounds_file=ccsnet_args["backgrounds"],
    ifos=ccsnet_args["ifos"],
    omicron_path=ccsnet_args["omicron_dir"],
    channels=ccsnet_args["channels"],
    frame_type=ccsnet_args["frame_type"], 
    sample_rate=ccsnet_args["sample_rate"],
    sample_duration=ccsnet_args["sample_duration"],
    fftlength=ccsnet_args["fftlength"],
    overlap=ccsnet_args["overlap"],
    device="cpu"
):
    
    glitch_file = glitch_merger(
        ifos=ifos,
        omicron_path=omicron_path,
        channels=channels
    )

    strain = get_background(
        seg=seg,
        ifos=ifos,
        frame_type=frame_type,
        channels=channels,
        sample_rate=sample_rate,
    )

    psds = psd_estimiater(
        ifos=ifos,
        strains=torch.tensor(strain).double(), 
        sample_rate=sample_rate,
        kernel_width=sample_duration,
        fftlength=fftlength,
        overlap=overlap
    ).to(device)

    with h5py.File(backgrounds_file, "a") as g:

        g1 = g.create_group(f"segments{seg_num:02d}")
        
        g1.create_dataset("strain", data=strain)
        g1.create_dataset("psd", data=psds)
        g1.attrs["start"] = seg[0]
        g1.attrs["end"] = seg[1]

        for key, item in h5_thang(glitch_file).h5_data().items():

            g1.create_dataset(key, data=item)

if __name__ == "__main__":

    segs = get_conincident_segs(
        ifos=ccsnet_args["ifos"],
        start=ccsnet_args["train_start"],
        stop=ccsnet_args["test_end"],
        state_flag=ccsnet_args["state_flag"]
    )

    for seg_num, seg in enumerate(segs):

        seg_duration = seg[1] - seg[0]
        if seg_duration <= 120:
            continue
        
        if seg_num <= 58:
            continue

        main(
            seg, 
            seg_num,
            omicron_path=ccsnet_args["omicron_dir"] / f"Segs_{seg_num:02d}",
        )