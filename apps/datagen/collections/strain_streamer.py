import h5py
import toml

import numpy as np
import logging

from pathlib import Path
from gwdatafind import find_urls
from argparse import ArgumentParser
from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityDict

from ccsnet.utils import h5_thang, args_control

parser = ArgumentParser()
parser.add_argument("-e", "--env", help="The env setting")
args = parser.parse_args()

ccsnet_args = args_control(
    args.env,
)

def background_collector(
    ifos,
    start,
    end,
    sample_rate,
    state_flag,
    frame_type,
    channels,
    background_path
):
    
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
        
        strain = []
        print(f"Collecting strain data from {start} to {end} at {channels}")
        for num in range(num_ifo):
        
            files = find_urls(
                site=f"{ifos[num][0]}",
                frametype=f"{ifos[num]}_{frame_type[num]}",
                gpsstart=start,
                gpsend=end,
                urltype="file",
            )
            
            strain.append(
                TimeSeries.read(
                    files,
                    f"{ifos[num]}:{channels[num]}",
                    start=start,
                    end=end,
                    nproc=8,
                    verbose=True
                ).resample(sample_rate).value
            )
        
        strain = np.stack(strain)
        
        # Fix enviroment problem to support PSD pre calculation 
        background_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(background_path, 'a') as g:
            
            g1 = g.create_group(f"segments{idx:02d}")
            
            g1.create_dataset("strain", data=strain)
            g1.attrs["start"] = start
            g1.attrs["end"] = end

if __name__ == "__main__":
    
#     # background_collector(
#     #     ifos = ccsnet_args["ifos"],
#     #     train_start = ccsnet_args["train_start"], # 1262254622,
#     #     train_stop = ccsnet_args["train_end"], # 1262686622,
#     #     sample_rate = ccsnet_args["sample_rate"], # 4096,
#     #     state_flag = ccsnet_args["state_flag"], # ["DCS-ANALYSIS_READY_C01", "DCS-ANALYSIS_READY_C01"],
#     #     frame_type = ccsnet_args["frame_type"], # ["HOFT_C01", "HOFT_C01"],
#     #     channels = ccsnet_args["channels"], # ["DCS-CALIB_STRAIN_CLEAN_C01", "DCS-CALIB_STRAIN_CLEAN_C01"],
#     #     background_path = Path(ccsnet_args["backgrounds"]) # "/home/hongyin.chen/Data/CCSNet/production/five_day_run/background.h5"
#     # )
    
    
    background_collector(
        ifos = ccsnet_args["ifos"],
        start = ccsnet_args["test_start"], # 1262254622,
        end = ccsnet_args["test_end"], # 1262686622,
        sample_rate = ccsnet_args["sample_rate"], # 4096,
        state_flag = ccsnet_args["state_flag"], # ["DCS-ANALYSIS_READY_C01", "DCS-ANALYSIS_READY_C01"],
        frame_type = ccsnet_args["frame_type"], # ["HOFT_C01", "HOFT_C01"],
        channels = ccsnet_args["channels"], # ["DCS-CALIB_STRAIN_CLEAN_C01", "DCS-CALIB_STRAIN_CLEAN_C01"],
        background_path = ccsnet_args["test_backgrounds"] # "/home/hongyin.chen/Data/CCSNet/production/five_day_run/background.h5"
    )
