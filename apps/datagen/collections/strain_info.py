import toml
import torch

from pathlib import Path

from ccsnet.omicron import glitch_merger, psd_estimiater
from ccsnet.utils import h5_thang
ARGUMENTS_FILE = "/home/hongyin.chen/anti_gravity/CCSNet/apps/train/trainer/arguments.toml"

ccsnet_arguments = toml.load(ARGUMENTS_FILE)

# To Do:
# Provide glitch info at each segments
# Provide PSD at each segment

# glitch_merger(
#     ifos=ccsnet_arguments["ifos"],
#     omicron_path=Path(ccsnet_arguments["omicron_output"]),
#     channels=ccsnet_arguments["channels"],
#     output_file=ccsnet_arguments["glitch_info"]
# )

background_info = h5_thang(ccsnet_arguments["backgrounds"])

psd_estimiater(
    ifos=ccsnet_arguments["ifos"],
    strains=torch.tensor(background_info.h5_data()["segments00/strain"]).double(),  # This part need to modify to segments-wise operation
    sample_rate=ccsnet_arguments["sample_rate"],
    kernel_width=ccsnet_arguments["sample_duration"],
    fftlength=ccsnet_arguments["fftlength"],
    overlap=ccsnet_arguments["overlap"],
    psd_path=Path(ccsnet_arguments["psd_files"])
)