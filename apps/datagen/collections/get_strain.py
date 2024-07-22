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
from ccsnet.omicron import glitch_merger, psd_estimiater, call_gwf_from_local

from run_omicron import get_conincident_segs

parser = ArgumentParser()
parser.add_argument("-e", "--env", help="The env setting")
args = parser.parse_args()

ccsnet_args = args_control(
    args.env,
)

logging.basicConfig(
    filemode='a',
    format="%(asctime)s %(name)s %(levelname)s:\t%(message)s",
    datefmt='%H:%M:%S',
    level=logging.NOTSET
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
    # for num in range(num_ifo):
    for i, ifo in enumerate(ifos):
    
        # Hard coded for fetching KAGRA data
        if ifo == "K1":
            files = call_gwf_from_local(
                ifo=ifo,
                start=seg[0],
                end=seg[1],
            )
        else:
            files = find_urls(
                site=f"{ifo[0]}",
                frametype=f"{ifo}_{frame_type[i]}",
                gpsstart=seg[0],
                gpsend=seg[1],
                urltype="file",
            )
        
        strain.append(
            TimeSeries.read(
                files,
                f"{ifo}:{channels[i]}",
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

    logging.info(f"Saving background to {backgrounds_file}")
    
    with h5py.File(backgrounds_file, "a") as g:

        g1 = g.create_group(f"segments{seg_num:02d}")
        
        g1.create_dataset("strain", data=strain)
        g1.create_dataset("psd", data=psds)
        g1.attrs["start"] = seg[0]
        g1.attrs["end"] = seg[1]

        for key, item in h5_thang(glitch_file).h5_data().items():

            g1.create_dataset(key, data=item)

if __name__ == "__main__":

    segs = ccsnet_args.get("ana_segs")

    if segs is None:

        segs = get_conincident_segs(
            ifos=ccsnet_args["ifos"],
            start=ccsnet_args["train_start"],
            stop=ccsnet_args["test_end"],
            state_flag=ccsnet_args["state_flag"]
        )

    for seg_num, seg in enumerate(segs):

        main(
            seg, 
            seg_num,
            omicron_path=ccsnet_args["omicron_dir"] / f"Segs_{seg_num:02d}",
        )