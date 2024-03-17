import toml
import torch
import h5py 

import numpy as np

import time
from tqdm import tqdm
from pathlib import Path
from ccsnet.arch import WaveNet
from ccsnet.utils import h5_thang
from torch.utils.data import Dataset, TensorDataset, DataLoader
from ml4gw.transforms import Whiten


def model_loader(
    num_ifos: int,
    sample_rate: int,
    # f_duration: int,
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

class CCSNet_Dataset(Dataset):

    def __init__(
        self, 
        signal, 
        n_ifos=2, 
        sample_rate=4096,
        sample_duration=3,
        device="cpu"
    ):

        # Get data type from https://pytorch.org/docs/stable/tensors.html
        self.signal = torch.FloatTensor(
            signal.reshape([-1, n_ifos, sample_duration*sample_rate])
        ).to(device)
    
    def __len__(self):
        
        return len(self.signal)
        
    def __getitem__(self, index):
        x = self.signal[index]

        return x, index
    
def test_data_loader(
    signal,
    n_ifos=2, 
    sample_rate=4096,
    sample_duration=3,
    batch_size: int = 1024,
    shuffle=False,
    device="cpu"
):

    dataset = CCSNet_Dataset(
        signal,
        n_ifos=n_ifos, 
        sample_rate=sample_rate,
        sample_duration=sample_duration,
        device=device
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class Streamer:

    def __init__(
        self,
        num_ifos: int,
        sample_rate: int,
        architecture,
        model_weights,
        fftlength,
        highpass,
        psds,
        device ='cpu'
    ):
        
        self.nn_model = model_loader(
            num_ifos=num_ifos,
            sample_rate = sample_rate,
            architecture = architecture,
            model_weights = model_weights,
            device = device
        )

        self.whiten_model = Whiten(
            fftlength,
            sample_rate,
            highpass,
        ).to(device)


        self.psds = psds.to(device)
        
    def __call__(
        self,
        dataloader,
    ):
        
        with torch.no_grad():
            for signal, index in dataloader:
                print("      True Excution")
                signal = self.whiten_model(
                    signal,
                    self.psds
                )
                
                pred = self.nn_model(signal)
                
                yield pred, index
            
        

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ccsnet_args = toml.load(Path.home() / "anti_gravity/CCSNet/apps/train/trainer/arguments.toml")
    
    test_dir = Path(ccsnet_args["test_dir"])
    out_dir = Path(ccsnet_args["output_dir"]) / "Analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    psd_path = test_dir / ccsnet_args["test_psds"]
    test_data_dir = test_dir / ccsnet_args["test_data"]
    model_weights = Path(ccsnet_args["test_model"])
    
    psds = torch.tensor(h5_thang(psd_path).h5_data(["psd"])["psd"]).double()
    
    selected_ccsn = Path("/home/hongyin.chen/anti_gravity/CCSNet/apps/train/ccsn.toml")
    
    bg_list = [
        "No_Glitch",
        "H1_Glitch",
        "L1_Glitch",
        "Combined"
    ]
    
    # Initialing
    
    ccsnet_streamer = Streamer(
        num_ifos=len(ccsnet_args["ifos"]),
        sample_rate=ccsnet_args["sample_rate"],
        architecture=WaveNet,
        model_weights=model_weights,
        fftlength=ccsnet_args["fftlength"],
        highpass=ccsnet_args["highpass"],
        psds=psds,
        device=device
    )
    
    for bg_mode in bg_list:
            
        for file in (sorted(test_data_dir.glob(f"{bg_mode}*.h5"))):

            # Data Loading
            with h5py.File(file, "r") as h:
                
                signal = h["Signal"][:]
            # Predicting

            data_loader = test_data_loader(
                signal,
                n_ifos=2, 
                sample_rate=ccsnet_args["sample_rate"],
                sample_duration=ccsnet_args["sample_duration"],
                batch_size=ccsnet_args["test_batch_size"],
                shuffle=False,
                device=device
            )

            # Streaming

            stream_out = ccsnet_streamer(data_loader)

            preds = []
            indexs = []

            # Output looping

            for pred, index in stream_out:
                
                preds.append(pred.cpu().detach().numpy())
                indexs.append(index.cpu().detach().numpy())
                
            preds = np.concatenate(preds).reshape([-1])
            indexs = np.concatenate(indexs).reshape([-1])

            # Output saving

            start = time.time()
            if file.name[:-3] == bg_mode:
                with h5py.File(out_dir / "Backgrounds.h5", "a") as g:
                    
                    h = g.create_group(f"{file.name[:-3]}")
                    
                    h.create_dataset(f"Signal", data=preds)
                    h.create_dataset(f"Index", data=indexs)
                
            else:
                
                mode_len = len(bg_mode) + 1
                with h5py.File(file, "r") as h:
                    
                    distance = h["distance"][:]
                distance = distance[indexs]
                with h5py.File(out_dir / f"{bg_mode}_Signals.h5", "a") as g:
                    
                    h = g.create_group(file.name[mode_len:-3])
                    
                    h.create_dataset(f"Signal", data=preds)
                    h.create_dataset(f"Index", data=indexs)
                    h.create_dataset(f"Distance", data=distance)
            

        # print(type(file.name))
    # project = "CCSNet_ADAM_03"
    # trained_weights = f"/home/hongyin.chen/Xperimental/CCSNet/sandbox/test_pub_RECOVER/Data/{project}/final_model.pt"
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DATA_DIR = Path.home() / "Xperimental/CCSNet/sandbox/test"
    # OUT_DIR = Path.home() / f"Xperimental/CCSNet/sandbox/test_pub_RECOVER/Data/{project}"
    # OUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # background_data = h5_thang(DATA_DIR / "raw_background.h5").h5_data()
    # injection_data = h5_thang(DATA_DIR / "raw_injection.h5").h5_data()
    # # trained_weights = "/home/hongyin.chen/Xperimental/CCSNet/ccsnet_dev/final_model"
    
    
    # bg_modes = [
    #     "h1_glitch",
    #     "l1_glitch",
    #     "bg",
    #     "glitches"
    # ]
    
    # nn_model = model_loader(
    #     2,
    #     sample_rate = 4096,
    #     architecture = WaveNet,
    #     model_weights = trained_weights,
    #     device = device
    # )
    
    # whiten_model = Whiten(
    #     fftlength,
    #     sample_rate,
    #     highpass
    # ).to("cuda")
    
    # psds = psds.to(device)
    # print(psds.dtype)

    # print("Initialized")
    # for mode, data in background_data.items():
    #     print(mode)
    #     background_loader = test_data_loader(
    #         torch.Tensor(data),
    #         batch_size=128,
    #         device=device
    #     )
        
    #     preds = []
    #     for _, x in enumerate(background_loader):
            
    #         x = x[0]
            
    #         x = whiten_model(
    #             x,
    #             psds
    #         )
            
    #         pred = nn_model(x)
    #         preds.append(pred.cpu().detach().numpy())
    #     # print(preds.shape)
    #     preds = np.concatenate(preds).reshape([-1])
        
    #     with h5py.File(OUT_DIR / f"background_result.h5", "a") as g:
            
    #         # h = g.create_group(f"{mode}")
    #         g.create_dataset(f"{mode}", data=preds)


    # for mode, data in tqdm(injection_data.items()):
        
    #     injection_loader = test_data_loader(
    #         torch.Tensor(data),
    #         batch_size=128,
    #         device=device
    #     )
        
    #     preds = []
    #     for _, x in enumerate(injection_loader):
            
    #         x = x[0]
            
    #         x = whiten_model(
    #             x,
    #             psds
    #         )
            
    #         pred = nn_model(x)
    #         preds.append(pred.cpu().detach().numpy())
            
    #     preds = np.concatenate(preds).reshape(-1)

    #     with h5py.File(OUT_DIR / f"injection_result.h5", "a") as g:
            
    #         # h = g.create_group(f"{mode}")
    #         g.create_dataset(f"{mode}", data=preds)
            
    
