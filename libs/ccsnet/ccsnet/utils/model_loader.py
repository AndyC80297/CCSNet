import torch

from pathlib import Path

def model_loader(
    num_ifos: int,
    # sample_rate: int,
    # f_duration: int,
    architecture,
    model_path: Path,
    device="cpu"
):

    nn_model = architecture(num_ifos)
    state_dict = torch.load(model_path, map_location="cpu")
    
    nn_model.load_state_dict(state_dict)
    nn_model.eval()
    
    nn_model.to(device)
    
    return nn_model

