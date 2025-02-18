# import h5py
import sys
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

parser = ArgumentParser()
parser.add_argument("-e", "--env", help="The env setting")
args = parser.parse_args()

ccsnet_arguments = args_control(
    args.env,
    saving=True
)

logging.basicConfig(
    filename= ccsnet_arguments["result_dir"] / "train.log",
    filemode='a',
    format="%(asctime)s %(name)s %(levelname)s:\t%(message)s",
    datefmt='%H:%M:%S',
    level=logging.NOTSET
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info("Booting CCSNet...")

def main(
    background_file = ccsnet_arguments["backgrounds"], 
    chosen_signals = ccsnet_arguments["train_siganls"],
    signals_dir = ccsnet_arguments["signals_dir"],
    max_iteration = ccsnet_arguments["max_iteration"], 
    batch_size = ccsnet_arguments["batch_size"], 
    steps_per_epoch = ccsnet_arguments["steps_per_epoch"], 
    sample_rate = ccsnet_arguments["sample_rate"], 
    sample_duration = ccsnet_arguments["sample_duration"], 
    off_set = ccsnet_arguments["off_set"],
    ifos = ccsnet_arguments["ifos"], 
    fftlength = ccsnet_arguments["fftlength"], 
    overlap = ccsnet_arguments["overlap"],
    highpass = ccsnet_arguments["highpass"], 
    model=WaveNet, 
    pretrained_model=None, 
    weight_decay=ccsnet_arguments["weight_decay"], 
    learning_rate=ccsnet_arguments["learning_rate"], 
    outdir: Path=ccsnet_arguments["result_dir"], 
    signal_chopping=ccsnet_arguments["signal_chopping"],
    val_count=ccsnet_arguments["val_count"],
    val_batch=ccsnet_arguments["val_batch"],
    device: str="cuda", 
):
    """
    Basically this founction server as an interface of how the data steams
    to the model. It acts like a placeholder that aranges its arguments to
    different functions for calling, iterate, or excution. 
    """

    training_segment = "segments00"
    psd = h5_thang(background_file).h5_data([f"{training_segment}/psd"])[f"{training_segment}/psd"]
    psds = torch.cuda.DoubleTensor(psd)

    signals_dict = load_h5_as_dict(
        chosen_signals,
        signals_dir
    )

    background_sampler = BackGroundDisplay(
        ifos,
        background_file = background_file,
        segment=training_segment,
        sample_rate = sample_rate,
        sample_duration = sample_duration,
        outdir=outdir
    )
    
    signal_sampler = Injector(
        ifos=ifos, 
        signals_dict = signals_dict,
        sample_rate = sample_rate,
        sample_duration = sample_duration,
        psds=psds,
        fftlength=fftlength,
        overlap=overlap,
        outdir=outdir,
        signal_chopping=signal_chopping,
        batch_size = batch_size,
        steps_per_epoch = steps_per_epoch,
        off_set=off_set,
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
        signal_chopping=signal_chopping,
        psds=psds,
        fftlength=fftlength,
        overlap=overlap,
        sample_rate=sample_rate, 
        count=val_count,
        batch_size=val_batch,
        sample_duration=sample_duration, 
        max_iteration=max_iteration,
        output_dir=outdir
    )

    Tachyon(
        architecture=model, 
        background_sampler=background_sampler, 
        signal_sampler=signal_sampler,
        noise_glitch_dist=[0, 0.375, 0.375, 0.25],
        signal_glitch_dist=None,
        validation_scheme=validation_scheme,
        whiten_model=whiten_model,
        psds=psds,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        max_iteration=max_iteration,
        num_ifo=len(ifos),
        pretrained_model=pretrained_model,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        device=device,
        outdir=outdir
    )


if __name__ == "__main__":
    data = main()
    