import toml
import h5py

from pathlib import Path

from ccsnet.utils import h5_thang
from ccsnet.omicron import glitch_merger
ARGUMENTS_FILE = "/home/hongyin.chen/anti_gravity/CCSNet/apps/train/trainer/arguments.toml"

ccsnet_args = toml.load(ARGUMENTS_FILE)
data_dir = Path(ccsnet_args["data_dir"])
test_dir = Path(ccsnet_args["test_dir"])
glitch_info = h5_thang(data_dir /ccsnet_args["glitch_info"])

OMICRON_DIR = Path('/home/hongyin.chen/Xperimental/CCSNet/sandbox/Triggers/Omicron/Omicron_out/Raw_test_01')


frame = 'CALIB_STRAIN_CLEAN_C01'

# h1_glitch_file = OMICRON_DIR / f'Raw_test_01/H1/merge/H1:DCS-{frame}/H1-DCS_{frame}_OMICRON-1262911490-32764.h5'
# l1_glitch_file = OMICRON_DIR / f'Raw_test_01/L1/merge/L1:DCS-{frame}/L1-DCS_{frame}_OMICRON-1262911490-32764.h5'

# print(h1_glitch_file)
# print(l1_glitch_file)



glitch_merger(
    ifos=ccsnet_args["ifos"],
    omicron_path=OMICRON_DIR,
    channels=ccsnet_args["channels"],
    output_file=test_dir / ccsnet_args["test_glitch_info"]
)