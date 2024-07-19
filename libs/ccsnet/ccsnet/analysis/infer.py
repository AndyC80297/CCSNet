import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader


class CCSNe_Dataset(Dataset):

    def __init__(
        self, 
        signal, 
        scaled_distance=None,
        n_ifos=2, 
        sample_rate=4096,
        sample_duration=3,
        device="cpu"
    ):

        # Get data type from https://pytorch.org/docs/stable/tensors.html
        self.signal = torch.FloatTensor(
            signal.reshape([-1, n_ifos, sample_duration*sample_rate])
        ).to(device)
        
        self.scaled_distance = scaled_distance
        if self.scaled_distance is not None:
            self.scaled_distance = torch.FloatTensor(
                scaled_distance.reshape([-1, 1])
            ).to(device)

    def __len__(self):
        
        return len(self.signal)
        
    def __getitem__(self, index):
        x = self.signal[index]

        if self.scaled_distance is not None:
            dis = self.scaled_distance[index]

            return x, dis, index

        return x, index

def test_data_loader(
    signal,
    n_ifos=2, 
    sample_rate=4096,
    sample_duration=3,
    batch_size: int = 1024,
    shuffle=False,
    device="cpu",
    scaled_distance=None,
):

    dataset = CCSNe_Dataset(
        signal,
        scaled_distance,
        n_ifos=n_ifos, 
        sample_rate=sample_rate,
        sample_duration=sample_duration,
        device=device
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



class Inference_Dataset(Dataset):

    def __init__(
        self, 
        signal, 
        scaled_distance=None,
        n_ifos=2, 
        sample_rate=4096,
        sample_duration=600,
        device="cpu"
    ):

        # Get data type from https://pytorch.org/docs/stable/tensors.html
        self.signal = torch.FloatTensor(
            signal.reshape([-1, n_ifos, sample_duration*sample_rate])
        ).to(device)
        
        self.scaled_distance = scaled_distance
        if self.scaled_distance is not None:
            self.scaled_distance = torch.FloatTensor(
                scaled_distance.reshape([-1, 1])
            ).to(device)

    def __len__(self):
        
        return len(self.signal)
        
    def __getitem__(self, index):
        x = self.signal[index]

        if self.scaled_distance is not None:
            dis = self.scaled_distance[index]

            return x, dis, index

        return x, index


def model_loader(
    num_ifos: int,
    architecture,
    model_weights,
    map_device = "cpu",
    device="cpu"
):
    
    nn_model = architecture(num_ifos)

    state_dict = torch.load(
        model_weights, 
        map_location=torch.device(map_device), 
        weights_only=True
    )
    
    nn_model.load_state_dict(state_dict)
    nn_model.eval()
    
    nn_model.to(device)
    
    return nn_model