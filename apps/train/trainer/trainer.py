# import h5py
import toml
import torch
import ml4gw
import logging
import numpy as np

from pathlib import Path
from argparse import ArgumentParser

from validator import Validator
from orchestrator import BackGroundDisplay, Injector, forged_dataloader

from ccsnet import train
from ccsnet.arch import WaveNet
from ccsnet.utils import h5_thang, args_control
from ccsnet.waveform import load_h5_as_dict
from ccsnet.train.train import Tachyon

from ml4gw.transforms import Whiten
from ml4gw.transforms.transform import FittableSpectralTransform

logging.basicConfig(level=logging.NOTSET)

parser = ArgumentParser()
parser.add_argument("-e", "--env", help="The env setting")
args = parser.parse_args()

ccsnet_arguments = args_control(
    args.env,
    saving=True
)

def main(
    background_file = ccsnet_arguments["backgrounds"], 
    chosen_signals = ccsnet_arguments["train_siganls"],
    signals_dir = ccsnet_arguments["signals_dir"],
    # init_distance = ccsnet_arguments["init_distance"],
    # glitch_info = ccsnet_arguments["glitch_info"], 
    max_iteration = ccsnet_arguments["max_iteration"], 
    batch_size = ccsnet_arguments["batch_size"], 
    steps_per_epoch = ccsnet_arguments["steps_per_epoch"], 
    sample_rate = ccsnet_arguments["sample_rate"], 
    sample_duration = ccsnet_arguments["sample_duration"], 
    ifos = ccsnet_arguments["ifos"], 
    fftlength = ccsnet_arguments["fftlength"], 
    overlap = ccsnet_arguments["overlap"],
    # psd_file = ccsnet_arguments["psds"], 
    highpass = ccsnet_arguments["highpass"], 
    model=WaveNet, 
    pretrained_model = None, 
    weight_decay = ccsnet_arguments["weight_decay"], 
    learning_rate = ccsnet_arguments["learning_rate"], 
    # outdir: Path = ccsnet_arguments["output_dir"], 
    val_sqrtnum = ccsnet_arguments["val_sqrtnum"],
    device: str = "cuda", 
):
    """
    Basically this founction server as an interface of how the data steams
    to the model. It acts like a placeholder that aranges its arguments to
    different functions for calling, iterate, or excution. 
    """

    # psd = h5_thang(psd_file).h5_data()["psd"]
    # psds = torch.cuda.DoubleTensor(psd)

    signals_dict = load_h5_as_dict(
        chosen_signals,
        signals_dir
    )

    background_sampler = BackGroundDisplay(
        ifos,
        background_file = background_file,
        # glitch_info = glitch_info,
        sample_rate = sample_rate,
        sample_duration = sample_duration,
        # outdir=outdir
    )
    
    # max_distance = {}

    # for name in signals_dict.keys():
    #     max_distance[name] = init_distance

    
    signal_sampler = Injector(
        ifos=ifos,
        # background=injection_siganl, 
        signals_dict = signals_dict,
        init_distance = init_distance,
        sample_rate = sample_rate,
        sample_duration = sample_duration,
        psds=psds,
        fftlength=fftlength,
        overlap=overlap,
        # outdir = outdir,
        batch_size = batch_size,
        steps_per_epoch = steps_per_epoch,
    )


    whiten_model = Whiten(
        fftlength,
        sample_rate,
        highpass
    ).to("cuda")


    validation_scheme = Validator(
        ifos=ifos,
        signals_dict=signals_dict,
        chosen_signals=chosen_signals,
        psds=psds,
        fftlength=fftlength,
        overlap=overlap,
        # batch_size=val_batch_size,
        sample_rate=sample_rate, 
        sqrtnum=val_sqrtnum,
        sample_duration=sample_duration, 
        max_iteration=max_iteration,
        # output_dir=outdir
    )

    Tachyon(
        architecture=model, 
        background_sampler=background_sampler, 
        signal_sampler=signal_sampler,
        # max_distance=max_distance,
        noise_glitch_dist = [0, 0.375, 0.375, 0.25],
        signal_glitch_dist = [0.7247, 0.09, 0.17, 0.0153],
        # noise_glitch_dist = [1, 0, 0, 0],
        # signal_glitch_dist = [0.25, 0.25, 0.25, 0.25],
        validation_scheme=validation_scheme,
        whiten_model=whiten_model,
        psds = psds,
        batch_size = batch_size,
        steps_per_epoch = steps_per_epoch,
        max_iteration = max_iteration,
        num_ifo = len(ifos),
        pretrained_model = pretrained_model,
        weight_decay = weight_decay,
        learning_rate = learning_rate,
        device = "cuda",
        # outdir= outdir
    )



if __name__ == "__main__":
    data = main()
    