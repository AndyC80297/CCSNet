import h5py
import torch 

import numpy as np
from pathlib import Path

from ml4gw.transforms.transform import FittableSpectralTransform

glitch_keys = [
    'time', 
    'frequency', 
    'tstart', 
    'tend', 
    'fstart', 
    'fend', 
    'snr', 
    'q', 
    'amplitude', 
    'phase'
]

def glitch_merger(
    ifos,
    omicron_path: Path,
    channels,
    output_file,
    glitch_keys=glitch_keys
):


    for i, ifo in enumerate(ifos):

        gltich_dir = omicron_path / f"{ifo}/trigger_output/merge/{ifo}:{channels[i]}"

        h5_name = {}
        for key in glitch_keys:

            h5_name[key] = []   

        for file in sorted(gltich_dir.glob("*.h5")):

            with h5py.File(file, "r") as h:
                
                for key in glitch_keys:
                    
                    h5_name[key].append(h["triggers"][key])
                    
        for key in glitch_keys:
            h5_name[key] = np.concatenate(h5_name[key])
            
        with h5py.File(output_file, "a") as g:
            
            g1 = g.create_group(ifo)
            
            for key in glitch_keys:
                g1.create_dataset(key, data=h5_name[key])
                
                
def psd_estimiater(
    ifos: list,
    strains: torch.tensor,
    sample_rate: int,
    kernel_width: int,
    fftlength: int,
    overlap: int,
    # psd_path
) -> torch.Tensor:
    """_summary_

    Args:
        ifos (list): Detectors.
        strains (torch.tensor): 
        sample_rate (int): Sampling rate of the strain data.
        kernel_width (int): Sampling window in secound.
        fftlength (int): 
        overlap (int): 

    Returns:
        torch.Tensor: 
    """
    
    num_channels = len(ifos)
    psds = torch.empty([num_channels, int((sample_rate*kernel_width)/2) +1]).double()
    
    spec_trans = FittableSpectralTransform()
    
    for i in range(num_channels):
        psds[i, :] = spec_trans.normalize_psd(
            strains[i],
            sample_rate=sample_rate,
            num_freqs=int((sample_rate*kernel_width)/2) +1,
            fftlength=fftlength,
            overlap=overlap,
        )

    return psds