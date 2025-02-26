import os
import sys
import shutil
import typing
import logging

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
    envs_file,
    project=project,
    test_segment=None,
    saving=False
):

    envs = dotenv_values(envs_file)

    code_base = Path(envs["CODE_BASE"])
    
    if project is not None:
        project = Path(project)
        result_dir = Path(envs["RESULT_DIR"]) / project
    else: 
         result_dir = Path(envs["RESULT_DIR"]) / "Test_run"
    result_dir.mkdir(parents=True, exist_ok=True)
    

    # Argument control
    if test_segment is not None:
        ccsnet_args_file = result_dir.parents[0] / "arguments.toml"
    else:
        ccsnet_args_file = code_base / "apps/arguments.toml"
    
    ccsnet_args = toml.load(ccsnet_args_file)
    
    ccsnet_args["result_dir"] = result_dir
    ccsnet_args["train_siganls"] = code_base / "apps/train/ccsn.toml"
    ccsnet_args["test_signals"] = code_base / "apps/tests/test_ccsn.toml"
    
    # Data Control
    data_dir = Path(envs["DATA_PATH"])
    data_dir = Path(data_dir)
    ccsnet_args["data_dir"] = data_dir
    test_data_dir = data_dir / "Tests"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    ccsnet_args["test_data_dir"] = test_data_dir
    
    ccsnet_args["backgrounds"] = data_dir / ccsnet_args["backgrounds"]
    ccsnet_args["signals_dir"] = data_dir / ccsnet_args["signals_dir"]
    ccsnet_args["omicron_dir"] = data_dir / ccsnet_args["omicron_dir"]
    
    ccsnet_args["test_backgrounds"] = test_data_dir / ccsnet_args["test_backgrounds"]
    ccsnet_args["test_siganl_parameter"] = test_data_dir / ccsnet_args["test_siganl_parameter"]
    
    if test_segment is not None:

        if len(Path(test_segment).parents) == 1:

            test_seg_data_dir = test_data_dir / Path(test_segment)
            test_seg_data_dir.mkdir(parents=True, exist_ok=True)
        else:
            
            test_seg_data_dir = test_data_dir / Path(test_segment).parents[0]
        ccsnet_args["sampled_background"] = test_seg_data_dir / ccsnet_args["sampled_background"]
        ccsnet_args["test_psd_seg"] = test_seg_data_dir / ccsnet_args["test_psd_seg"] # ccsnet_args["saved_psd"]
        
    ccsnet_args["test_model"] = result_dir / ccsnet_args["test_model"]

    if test_segment is not None:
        test_result_dir = result_dir / "Tests" / test_segment
        test_result_dir.mkdir(parents=True, exist_ok=True)
        ccsnet_args["test_result_dir"] = test_result_dir
    
    if saving:

        if project is None:

            import sys

            sys.exit("Project is None!")


        if len(project.parents) < 2:

            import sys

            sys.exit("Please provide run number!")

        args_saving(ccsnet_args_file, result_dir.parents[0])
        
        
    import sys
    logging.basicConfig(
        filename= ccsnet_args["test_result_dir"] / "test.log",
        filemode="a",
        format="%(asctime)s %(name)s %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
        level=logging.NOTSET
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info(f"Using argument file at {ccsnet_args_file}")
    return ccsnet_args