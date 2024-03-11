import toml
import torch
import h5py 

import numpy as np

from tqdm import tqdm
from pathlib import Path
from ccsnet.arch import WaveNet
from ccsnet.utils import h5_thang
from torch.utils.data import TensorDataset, DataLoader
from ml4gw.transforms import Whiten




def model_loader(
    num_ifos: int,
    sample_rate: int,
    # f_duration: int,
    architecture,
    model_weights,
    device='cpu'
):

    nn_model = architecture(num_ifos)

    state_dict = torch.load(
        model_weights, 
        map_location=torch.device('cpu'), 
        weights_only=True
    )
    
    nn_model.load_state_dict(state_dict)
    nn_model.eval()
    
    nn_model.to(device)
    
    return nn_model

def test_data_loader(
    signal,
    batch_size: int = 32,
    device="cpu"
):
    
    dataset = TensorDataset(
        signal.to(device),
    )
    
    return DataLoader(dataset, batch_size=batch_size)





if __name__ == '__main__':
    
    ccsnet_args = toml.load(Path.home() / "anti_gravity/CCSNet/apps/train/trainer/arguments.toml")
    
    fftlength = ccsnet_args["fftlength"]
    sample_rate = ccsnet_args["sample_rate"]
    highpass = ccsnet_args["highpass"]
    test_dir = Path(ccsnet_args["test_dir"])
    psd_path = test_dir / ccsnet_args["test_psds"]
    psds = torch.tensor(h5_thang(psd_path).h5_data(["psd"])["psd"]).double()
    selected_ccsn = toml.load(ccsnet_args["chosen_signals"])
    
    
    
    project = "CCSNet_ADAM_03"
    trained_weights = f"/home/hongyin.chen/Xperimental/CCSNet/sandbox/test_pub_RECOVER/Data/{project}/final_model.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = Path.home() / "Xperimental/CCSNet/sandbox/test"
    OUT_DIR = Path.home() / f"Xperimental/CCSNet/sandbox/test_pub_RECOVER/Data/{project}"
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    
    background_data = h5_thang(DATA_DIR / "raw_background.h5").h5_data()
    injection_data = h5_thang(DATA_DIR / "raw_injection.h5").h5_data()
    # trained_weights = "/home/hongyin.chen/Xperimental/CCSNet/ccsnet_dev/final_model"
    
    
    bg_modes = [
        "h1_glitch",
        "l1_glitch",
        "bg",
        "glitches"
    ]
    
    nn_model = model_loader(
        2,
        sample_rate = 4096,
        architecture = WaveNet,
        model_weights = trained_weights,
        device = device
    )
    
    whiten_model = Whiten(
        fftlength,
        sample_rate,
        highpass
    ).to("cuda")
    
    psds = psds.to(device)
    print(psds.dtype)

    print("Initialized")
    for mode, data in background_data.items():
        print(mode)
        background_loader = test_data_loader(
            torch.Tensor(data),
            batch_size=128,
            device=device
        )
        
        preds = []
        for _, x in enumerate(background_loader):
            
            x = x[0]
            
            x = whiten_model(
                x,
                psds
            )
            
            pred = nn_model(x)
            preds.append(pred.cpu().detach().numpy())
        # print(preds.shape)
        preds = np.concatenate(preds).reshape([-1])
        
        with h5py.File(OUT_DIR / f"background_result.h5", "a") as g:
            
            # h = g.create_group(f"{mode}")
            g.create_dataset(f"{mode}", data=preds)


    for mode, data in tqdm(injection_data.items()):
        
        injection_loader = test_data_loader(
            torch.Tensor(data),
            batch_size=128,
            device=device
        )
        
        preds = []
        for _, x in enumerate(injection_loader):
            
            x = x[0]
            
            x = whiten_model(
                x,
                psds
            )
            
            pred = nn_model(x)
            preds.append(pred.cpu().detach().numpy())
            
        preds = np.concatenate(preds).reshape(-1)

        with h5py.File(OUT_DIR / f"injection_result.h5", "a") as g:
            
            # h = g.create_group(f"{mode}")
            g.create_dataset(f"{mode}", data=preds)
