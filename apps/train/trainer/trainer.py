# import h5py
import toml
import torch
import ml4gw
import logging
import numpy as np

from pathlib import Path

from validator import Validator
from orchestrator import BackGroundDisplay, Injector, forged_dataloader

from ccsnet import train
from ccsnet.arch import WaveNet
from ccsnet.utils import h5_thang
from ccsnet.waveform import load_h5_as_dict
from ccsnet.train.train import Tachyon

from ml4gw.transforms import Whiten
from ml4gw.transforms.transform import FittableSpectralTransform

ARGUMENTS_FILE = "/home/hongyin.chen/anti_gravity/CCSNet/apps/train/trainer/arguments.toml"

ccsnet_arguments = toml.load(ARGUMENTS_FILE)

logging.basicConfig(level=logging.NOTSET)


bg_file_dict = h5_thang(ccsnet_arguments["backgrounds"]).h5_data()
psd = h5_thang(ccsnet_arguments["psd_files"]).h5_data()["psd"]
# signals_dict = h5_thang(ccsnet_arguments["signals"]).h5_data(["signals"]) # This data is in shape (500, 2, 16384)
signals_dict = load_h5_as_dict(
    ccsnet_arguments["chosen_signals"],
    ccsnet_arguments["signals"]
)

# Pass in arguments
# Function starts at here 
def main(
    background_file = ccsnet_arguments["backgrounds"], 
    signals_dict = signals_dict, 
    init_distance = ccsnet_arguments["init_distance"],
    glitch_info = ccsnet_arguments["glitch_info"], 
    max_iteration = ccsnet_arguments["max_iteration"], 
    batch_size = ccsnet_arguments["batch_size"], 
    steps_per_epoch = ccsnet_arguments["steps_per_epoch"], 
    sample_rate = ccsnet_arguments["sample_rate"], 
    sample_duration = ccsnet_arguments["sample_duration"], 
    ifos = ccsnet_arguments["ifos"], 
    fftlength = ccsnet_arguments["fftlength"], 
    psd = psd, 
    highpass = ccsnet_arguments["highpass"], 
    model=WaveNet, 
    pretrained_model = None, 
    weight_decay = ccsnet_arguments["weight_decay"], 
    learning_rate = ccsnet_arguments["learning_rate"], 
    outdir: Path = Path(ccsnet_arguments["output_dir"]), 
    val_batch_size = ccsnet_arguments["val_batch_size"],
    val_sqrtnum = ccsnet_arguments["val_sqrtnum"],
    device: str = "cuda", 
):
    """
    Basically this founction server as an interface of how the data steams
    to the model. It acts like a placeholder that aranges its arguments to
    different functions for calling, iterate, or excution. 
    """
    
    
    num_ifos = len(ifos)
    ### Consider to add a wapper to control the input type and output behavior

    # Load data to memory

    # Sampling & Windowing
    ### Buffer setting 
    ### Try reading mutlipule ccsn hdf file at once 
    ### By providing names and index
    
    background_sampler = BackGroundDisplay(
        ifos,
        background_file = background_file,
        glitch_info = glitch_info,
        sample_rate = sample_rate,
        sample_duration = sample_duration,
    )
    
                                    # # Move this part in to training loop
                                    # glitch_signal, glitch_target = background_sampler(
                                    #     batch_size = batch_size,
                                    #     steps_per_epoch = steps_per_epoch,
                                    #     sample_factor = 1/2,
                                    #     glitch_dist = [0, 0.25, 0.25, 0.5],
                                    #     choice_mask = [0, 1, 2, 3],
                                    #     glitch_offset = 0.9,
                                    #     target_value = 0
                                    # )

                                    # injection_siganl, injection_target = background_sampler(
                                    #     batch_size = batch_size,
                                    #     steps_per_epoch = steps_per_epoch,
                                    #     sample_factor = 1/2,
                                    #     glitch_dist = [0.7247, 0.09, 0.17, 0.0153],
                                    #     choice_mask = [0, 1, 2, 3],
                                    #     glitch_offset = 0.9,
                                    #     target_value = 1
                                    # )

    max_distance = {}

    for name in signals_dict.keys():
        max_distance[name] = init_distance

    
    signal_sampler = Injector(
        ifos=ifos,
        # background=injection_siganl, 
        signals_dict=signals_dict,
        sample_rate = sample_rate,
        sample_duration = sample_duration
    )

                                    # injected_siganl = signal_sampler(
                                    #     background=injection_siganl,
                                    #     max_distance=max_distance
                                    # )

    whiten_model = Whiten(
        fftlength,
        sample_rate,
        highpass
    ).to("cuda")

                                    # training_loader = forged_dataloader(
                                    #     inputs = [glitch_signal, injected_siganl],
                                    #     targets = [glitch_target, injection_target],
                                    #     batch_size = batch_size
                                    # )


    valdation_scheme = Validator(
        ifos=ifos,
        signals_dict=signals_dict,
        batch_size=val_batch_size,
        sample_rate=sample_rate, 
        sqrtnum=val_sqrtnum,
        sample_duration=sample_duration, 
    )

    psds = torch.cuda.FloatTensor(psd)

    Tachyon(
        architecture=model, 
        background_sampler=background_sampler, 
        signal_sampler=signal_sampler,
        max_distance=max_distance,
        noise_glitch_dist = [0, 0.25, 0.25, 0.5],
        signal_glitch_dist = [0.7247, 0.09, 0.17, 0.0153],
        # validator=validator,
        whiten_model=whiten_model,
        psds = psds,
        batch_size = batch_size,
        steps_per_epoch = steps_per_epoch,
        max_iteration = max_iteration,
        num_ifo = num_ifos,
        pretrained_model = pretrained_model,
        weight_decay = weight_decay,
        learning_rate = learning_rate,
        device = "cuda",
        outdir= outdir
    )

    
    # Remeber to pass the training and validation method to one function loop s
    # Make sure that we can pass the training and validation function to the training loop 
    # While in the training loop create a "cache" that saves the current training iterations and steps 
    # Also make sure that this variables can be acess by any file in order to make on training loop variable checks

if __name__ == "__main__":
    data = main()
    

