import toml
import torch

import numpy as np
from ccsnet.arch import WaveNet
from ml4gw.transforms import SnrRescaler

# ARGUMENTS_FILE = "/home/hongyin.chen/anti_gravity/CCSNet/apps/train/trainer/arguments.toml"

# ccsnet_arguments = toml.load(ARGUMENTS_FILE)

# Consider Combining BackGroundDisplay and Injector
class BackGroundDisplay:
    
    def __init__(
        self,
        background,
        max_iteration, 
        batch_size,
        sample_rate,
        sample_duration,
        num_ifos
    ):
        
        self.background = torch.Tensor(background)
        self.max_iteration = max_iteration
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.num_ifos = num_ifos
        
        self.kernel_length = sample_rate * sample_duration
        self.sample_data = int(max_iteration*batch_size/2)
        
    def foward(
        self,
        sample_method = None
    ):

        X = torch.empty((self.sample_data, self.num_ifos, self.kernel_length))

        idxs = np.random.randint(
            0, 
            self.background.shape[-1] - self.kernel_length, 
            self.sample_data
        )

        for i, idx in enumerate(idxs):
            X[i, :, :] = self.background[:, idx: idx + self.kernel_length]
            
        targets = torch.full((self.sample_data,), 0)
        
        return X, targets

class Injector:

    def __init__(
        self,
        background,
        signals,
        max_iteration, 
        batch_size,
        sample_rate,
        sample_duration,
        num_ifos,
        highpass
    ):

        self.background = torch.Tensor(background)
        
        self.max_iteration = max_iteration
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.num_ifos = num_ifos
        
        self.kernel_length = sample_rate * sample_duration
        self.sample_data = int(max_iteration*batch_size/2)
        self.signals = torch.Tensor(signals[:self.sample_data, :, :])
        self.highpass = highpass
        
    def foward(
        self,
        sample_method = None
    ):
    
    
        X = torch.empty((self.sample_data, self.num_ifos, self.kernel_length))

        idxs = np.random.randint(
            0, 
            self.background.shape[-1] - self.kernel_length, 
            self.sample_data
        )

        for i, idx in enumerate(idxs):
            X[i, :, :] = self.background[:, idx: idx + self.kernel_length]
        
        ### We have to change this to use pre-caulated psd
        rescaler = SnrRescaler(
            num_channels=self.num_ifos, 
            sample_rate = self.sample_rate,
            waveform_duration = self.sample_duration,
            highpass = self.highpass,
        )


        rescaler.fit(
            self.signals[0], 
            self.signals[1],
            fftlength=self.sample_duration,
            overlap=1,
        )

        rescaled_signals, target_snrs, rescale_factor = rescaler.forward(
            self.signals[:, :, 4096:12288]
        )
        
        X += rescaled_signals
        targets = torch.full((self.sample_data,), 1)
        return  X, targets
    
def forged_dataloader(
    inputs: list,
    targets: list,
    batch_size,
    pin_memory=False
):

    dataset = torch.utils.data.TensorDataset(
        torch.cat(inputs), 
        torch.cat(targets).view(-1, 1).to(torch.float)
    )
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)