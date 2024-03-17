import os
import sys
import shutil
import typing
import argparse
import toml
from dotenv import dotenv_values

from pathlib import Path

host = os.environ.get("HOSTNAME")
project = os.environ.get('PROJECT')
cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES")

def args_saving(
    ccsnet_args_file,
    output_dir
):

    destination = output_dir / ccsnet_args_file.name
    shutil.copyfile(ccsnet_args_file, destination)

def args_control(
    ccsnet_args_file,
    envs_file,
    project=project
):


    envs = dotenv_values(envs_file)
    data_dir = Path(envs["DATA_PATH"])
    output_dir = Path(envs["BASE_DIR"])

    training_dir = Path(data_dir)
    test_dir = training_dir / "Tests"
    output_dir = Path(output_dir) / project

    test_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ccsnet_args = toml.load(ccsnet_args_file)

    ccsnet_args["backgrounds"] = training_dir / ccsnet_args["backgrounds"]
    ccsnet_args["psds"] = training_dir / ccsnet_args["psds"]
    ccsnet_args["glitch_info"] = training_dir / ccsnet_args["glitch_info"]
    ccsnet_args["signals_dir"] = training_dir / ccsnet_args["signals_dir"]

    ccsnet_args["test_backgrounds"] = test_dir / ccsnet_args["test_backgrounds"]
    ccsnet_args["test_psds"] = test_dir / ccsnet_args["test_psds"]
    ccsnet_args["test_glitch_info"] = test_dir / ccsnet_args["test_glitch_info"]
    ccsnet_args["test_signals_dir"] = test_dir / ccsnet_args["test_signals_dir"]
    ccsnet_args["test_datasets"] = test_dir / ccsnet_args["test_datasets"]

    ccsnet_args["test_model"] = output_dir / ccsnet_args["test_model"]

    ccsnet_args["output_dir"] = output_dir
    args_saving(ccsnet_args_file, output_dir)
    
    return ccsnet_args