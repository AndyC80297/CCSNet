import toml


import configparser

from pathlib import Path

import h5py
import numpy as np
# from omicron.cli.process import main as omicron_main
from create_lcs import create_lcs

arg_dict = toml.load("/home/hongyin.chen/anti_gravity/anomaly_detection/Get_Glitch/parameters.toml")
project_dir = Path(arg_dict["project_dir"])

def omicron_control(
    start_time,
    end_time,
    project_dir: Path,
    ifos,
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
    verbose: bool = True,
    state_flag=None,
    mode="GW"
):

    """Parses args into a format compatible for Pyomicron,
    then launches omicron dag
    """

    # pyomicron expects some arguments passed via
    # a config file. Create that config file
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
        bash_file_path = project_dir / f"{ifo}/run_{ifo}.sh"
        cache_file = project_dir / ifo / "data_file.lcf"
        output_dir = project_dir / f"{ifo}" / "trigger_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
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