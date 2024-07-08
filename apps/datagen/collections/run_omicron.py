import h5py
import subprocess
import configparser

import numpy as np

from pathlib import Path
from gwpy.segments import DataQualityDict
from concurrent.futures import ThreadPoolExecutor

from ccsnet.omicron import create_lcs


def run_bash(bash_file):

    subprocess.run(
        ["bash", f"{bash_file}"], 
    )

def omicron_control(
    ifos,
    start_time,
    end_time,
    project_dir: Path,
    # INI
    q_range,
    frequency_range,
    frame_type,
    channels,
    cluster_dt,
    sample_rate,
    chunk_duration,
    segment_duration,
    overlap_duration,
    mismatch_max,
    snr_threshold,
    # log_file: Path,
    verbose: bool = False,
    state_flag=None,
    mode="GW"
):

    """Parses args into a format compatible for Pyomicron,
    then launches omicron dag
    """

    # pyomicron expects some arguments passed via
    # a config file. Create that config file
    bash_files = []

    for i, ifo in enumerate(ifos):
        
        config = configparser.ConfigParser()
        section = mode
        config.add_section(section)

        config.set(section, "q-range", f"{q_range[0]} {q_range[1]}")
        config.set(section, "frequency-range", f"{frequency_range[0]} {frequency_range[1]}")
        config.set(section, "frametype", f"{ifo}_{frame_type[i]}")
        config.set(section, "channels", f"{ifo}:{channels[i]}")
        config.set(section, "cluster-dt", str(cluster_dt))
        config.set(section, "sample-frequency", str(sample_rate))
        config.set(section, "chunk-duration", str(chunk_duration))
        config.set(section, "segment-duration", str(segment_duration))
        config.set(section, "overlap-duration", str(overlap_duration))
        config.set(section, "mismatch-max", str(mismatch_max))
        config.set(section, "snr-threshold", str(snr_threshold))
        # in an online setting, can also pass state-vector,
        # and bits to check for science mode
        if state_flag != None:
            config.set(section, "state-flag", f"{ifo}:{state_flag}")

        config_file_path = project_dir / f"{ifo}/omicron_{ifo}.ini"
        bash_file_path = project_dir / f"{ifo}/run_omicron.sh"
        cache_file = project_dir / ifo / "data_file.lcf"
        output_dir = project_dir / f"{ifo}" / "trigger_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        bash_files.append(bash_file_path)

        # write config file
        with open(config_file_path, "w") as config_file:
            config.write(config_file)
            
        omicron_args = [
            f"omicron-process {section}",
            f"--gps {start_time} {end_time}",
            f"--ifo {ifo}",
            f"--config-file {str(config_file_path)}",
            f"--output-dir {str(output_dir)}",
            f"--cache-file {cache_file}",
            # f"--log-file {str(project_dir/ifo)}",
            "--verbose"
            # "request_disk=100M",
            # "--skip-gzip",
            # "--skip-rm",
        ]
        with open (bash_file_path, 'w') as rsh:
            for args in omicron_args:
                rsh.writelines(f"{args} \\\n")

    
    return bash_files

def get_conincident_segs(
    ifos:list,
    start:int,
    stop:int,
    state_flag:list,
):
    
    query_flag = []
    for i, ifo in enumerate(ifos):
        query_flag.append(f"{ifo}:{state_flag[i]}")

    flags = DataQualityDict.query_dqsegdb(
        query_flag,
        start,
        stop
    )

    segs = []

    for contents in flags.intersection().active.to_table():

        segs.append((contents["start"], contents["end"]))

    return segs



if __name__ == "__main__":

    from argparse import ArgumentParser
    from ccsnet.utils import args_control

    parser = ArgumentParser()
    parser.add_argument("-e", "--env", help="The env setting")
    args = parser.parse_args()
    
    ccsnet_args = args_control(
        args.env,
        saving=False
    )
    
    ana_segs = get_conincident_segs(
        ifos=ccsnet_args["ifos"],
        start=ccsnet_args["train_start"],
        stop=ccsnet_args["test_end"],
        state_flag=ccsnet_args["state_flag"]
    )

    bash_files = []
    
    for seg_count, (start, end) in enumerate(ana_segs):

        print(start, end)

        for ifo, frametype in zip(ccsnet_args["ifos"], ccsnet_args["frame_type"]):

            create_lcs(
                ifo=ifo,
                frametype=f"{ifo}_{frametype}",
                start_time=start,
                end_time=end,
                output_dir=ccsnet_args["omicron_dir"] / f"Segs_{seg_count:02d}",
                urltype="file"
            )

        bash_scripts = omicron_control(
            ifos=ccsnet_args["ifos"],
            start_time=start,
            end_time=end,
            project_dir=ccsnet_args["omicron_dir"] / f"Segs_{seg_count:02d}",
            
            q_range=ccsnet_args["q_range"],
            frequency_range=ccsnet_args["frequency_range"],
            frame_type=ccsnet_args["frame_type"],
            channels=ccsnet_args["channels"],
            cluster_dt=ccsnet_args["cluster_dt"],
            sample_rate=ccsnet_args["sample_rate"],
            chunk_duration=ccsnet_args["chunk_duration"],
            segment_duration=ccsnet_args["segment_duration"],
            overlap_duration=ccsnet_args["overlap_duration"],
            mismatch_max=ccsnet_args["mismatch_max"],
            snr_threshold=ccsnet_args["snr_threshold"],
        )

        for bash_script in bash_scripts:
            bash_files.append(bash_script)
    

    # for bash_file in bash_files:
    #     print(bash_file)
    with ThreadPoolExecutor(max_workers=2) as e:
    
        for bash_file in bash_files:
            e.submit(run_bash, bash_file)