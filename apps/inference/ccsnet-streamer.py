import h5py
import torch
import logging


from pathlib import Path
from argparse import ArgumentParser

from ml4gw.transforms import Whiten

from ccsnet.arch import WaveNet
from ccsnet.utils import h5_thang, args_control
from ccsnet.analysis import model_loader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser()
parser.add_argument("-e", "--env", help="The env setting")
# parser.add_argument("-s", "--seg", type=int, help="Testing segment")
# parser.add_argument("-s", "--start", type=int, help="Inference start time")
# parser.add_argument("-t", "--end", type=int, help="Inference end time")
args = parser.parse_args()


ccsnet_args = args_control(
    envs_file=args.env,
    saving=False
)


def strain_buffer(
    strain_source=Path("/home/hongyin.chen/Data/CCSNet/O4_run/Inference/strain.h5"),
    ifos=ccsnet_args["ifos"],
    channels=ccsnet_args["channels"],
    sample_rate=ccsnet_args["sample_rate"],
    device="cpu"
):
    
    num_ifos = len(ifos)
    strain_dict = h5_thang(strain_source).h5_data()
    strain = torch.zeros((num_ifos, sample_rate*60*10))

    for i, (ifo, channel) in enumerate(zip(ifos, channels)):
        strain[i, :] = torch.Tensor(strain_dict[f"{ifo}:{channel}"])
    # strain[1, :] = strain_dict["L1:GDS-CALIB_STRAIN_CLEAN"]
    
    return strain.to(device)
    
def main(
    strain_source=Path("/home/hongyin.chen/Data/CCSNet/O4_run/Inference/strain.h5"),
    background_file=ccsnet_args["backgrounds"],
    ifos=ccsnet_args["ifos"],
    channels=ccsnet_args["channels"],
    sample_rate=ccsnet_args["sample_rate"],
    fftlength=ccsnet_args["fftlength"],
    highpass=ccsnet_args["highpass"],
    architecture=WaveNet,
    sample_duration=ccsnet_args["sample_duration"],
    model_weights=ccsnet_args["test_model"],
    stride=ccsnet_args["stride"], 
    # stride=1,
    map_device="cpu",
    device="cpu",
):
    
    # Hard-coded
    bg_h5 = h5_thang(background_file)
    psds = torch.tensor(bg_h5.h5_data([f"segments11/psd"])[f"segments11/psd"]).double()
    num_ifos = len(ifos)

    strain = strain_buffer(
        strain_source=strain_source,
        ifos=ifos,
        channels=channels,
        sample_rate=sample_rate,
        device=device,
    )

    whiten_model = Whiten(
        fftlength,
        sample_rate,
        highpass,
    ).to(device)

    nn_model = model_loader(
        num_ifos=num_ifos,
        architecture = architecture,
        model_weights=model_weights,
        map_device=map_device,
        device = device
    )

    ticks = strain.shape[-1] / sample_rate
    total_window = int((ticks - sample_duration) / stride) + 1

    stream_value = torch.zeros((total_window), device=device)
    for i in range(total_window):

        shifter = int(i*sample_rate*stride)
        X = strain[:, shifter: shifter + sample_duration*sample_rate]
        X = whiten_model(X, psds)
        stream_value[i] = nn_model(X)

        print(i, stream_value[i])

    with h5py.File("/home/hongyin.chen/Outputs/ReConfiged_CCSNet/O4_run/first_test_run/Inference/stream.h5", "w") as g:

        g.create_dataset("stream", data=stream_value.cpu().detach().numpy())


    
if __name__ == "__main__":

    main()
    