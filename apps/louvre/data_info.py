import toml

import matplotlib as mpl

from pathlib import Path
from gwpy.segments import DataQualityDict

from argparse import ArgumentParser
from ccsnet.utils import args_control

# Background active segment
# PSDs
# CCSN signals
        # Time domain, Frequency domain
# Glitch 
        # Q-transform spectrogram
    

mpl.rcParams['figure.dpi'] = 500
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['hatch.linewidth'] = 3.0 

def plot_active_segments(
    train_start,
    train_end,
    test_segs: list,
    ana_end,
    ifos,
    ifo_state_flags,
    out_dir: Path,
    glitch_start=None,
    figname: str = "Active_segments.png",
    linewidth=3,
    transparent=True,
):
    
    out_dir.mkdir(parents=True, exist_ok=True)
    coin_seg = DataQualityDict()
    state_flags = []
    
    for ifo, state_flag in zip(ifos, ifo_state_flags):
        state_flags.append(f"{ifo}:{state_flag[:-2]}")
    
    flag_start = train_start
    if glitch_start != None:
        flag_start = glitch_start

    flags = DataQualityDict.query_dqsegdb(
        state_flags,
        flag_start,
        ana_end
    )

    # plt = flags.plot(
    #     "name", 
    #     figsize=(15, 2), 
    #     xlabel="GPSTime (s)", 
    #     title="Active segments"
    # )
    # print("Train End", train_end)
    flags = flags.intersection()
    # print(flags.active.to_table())
    flags.name = "H1&L1"
    coin_seg["H1&L1:ANALYSIS_READY_C01"] = flags
    
    plt = coin_seg.plot(
        "name", 
        figsize=(10, 1), 
        xlabel="GPSTime (s)", 
        title="Coincident Active segments"
    )
    
    ax = plt.gca()
    
    if glitch_start != None:
        
        ax.axvline(glitch_start, color="black", lw=linewidth, ls="-")
        ax.axvline(train_end, color="black", lw=linewidth, ls="-")
    
    ax.axvline(train_start, color="blue", lw=linewidth, ls="-")
    ax.axvline(train_end, color="blue", lw=linewidth, ls="-")
    # ax.axvspan(train_start, train_end, hatch='x', edgecolor="blue", fill=False, linewidth=linewidth)

    for start, end in test_segs:

        ax.axvline(start, color="#ee33ff", lw=linewidth, ls="-")
        ax.axvline(end, color="#ee33ff", lw=linewidth, ls="-")
        # ax.axvspan(start, end, hatch='x', edgecolor="#ff8b0f", fill=False, linewidth=linewidth)
    
    ax.get_axisbelow()
    ax.ticklabel_format(
        axis='x', 
        useOffset=False, 
        style="plain"
    )
    ax.grid(False, axis='both')
    
    plt.savefig(
        out_dir / figname, 
        transparent=transparent, 
        bbox_inches='tight'
    )
    
    
if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("-e", "--env", help="The env setting")
    args = parser.parse_args()
    
    ccsnet_args = args_control(
        args.env,
        saving=False
    )

    test_segs  = [
        (1262688611, 1262705801),
        (1262746233, 1262762284),
        (1262919719, 1262944254),
    ]

    plot_active_segments(
        train_start=ccsnet_args["train_start"],
        train_end=ccsnet_args["train_end"],
        test_segs=test_segs,
        ana_end=ccsnet_args["test_end"],
        ifos=ccsnet_args["ifos"],
        ifo_state_flags=ccsnet_args["state_flag"],
        out_dir=Path(ccsnet_args["data_dir"]) / "Louvre",
        figname="CCSNet_Active_segments.png",
        transparent=False
    )