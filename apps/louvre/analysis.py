# Loss
# Distribution
# TPR v.s. distance
# TPR v.s. SNR?
# FAR ?



import toml
import h5py

import numpy as np


from pathlib import Path


from ccsnet.utils import h5_thang

from bokeh import palettes
from bokeh.models import Span
from bokeh.plotting import figure, show
from bokeh.models import CategoricalColorMapper, Legend
from bokeh.io import output_notebook, output_file, push_notebook


ccsnet_args = toml.load("/home/hongyin.chen/anti_gravity/CCSNet/apps/train/trainer/arguments.toml")

ana_dir = Path(ccsnet_args["output_dir"]) / "Analysis"
test_signals_dict = toml.load(ccsnet_args["test_signals"])

the_louvre = Path("/home/hongyin.chen/Outputs/CCSNet_Out_dir/recover_runs_ADAMW/Model_01_Result/Louvre")

noise_list = [
    "No_Glitch",
    "H1_Glitch",
    "L1_Glitch",
    "Combined"
]

inj_dict = {}

for key, items in test_signals_dict.items():
    
    dict_item = []
    for item in items:
        
        dict_item.append(f"{key}_{item}")
    
    inj_dict[key] = dict_item


signal_tag = []
signal_keys = []
signal_dis_keys = []
snrs = ccsnet_args["snr_distro"]

for snr in snrs:
    
    for key, items in test_signals_dict.items():
        
        for item in items:
            
            signal_tag.append(f"{key}_{item}_{snr:02d}")
            signal_keys.append(f"{key}_{item}_{snr:02d}/Signal")
            signal_dis_keys.append(f"{key}_{item}_{snr:02d}/Distance")


noise_data = [noise + "/Signal" for noise in noise_list]
bg_data = h5_thang(ana_dir / "Backgrounds.h5").h5_data(noise_data)

sig_dis_data = h5_thang(ana_dir / "Combined_Signals.h5").h5_data(signal_keys)
sig_dis_data = h5_thang(ana_dir / "No_Glitch_Signals.h5").h5_data(signal_keys)


bg_colors = palettes.Category10[10]

inj_colors = list(palettes.Category10[10])
for color in palettes.Colorblind[3]:
    inj_colors.append(color)

line_patten = [
    "solid", 
    "dotted", 
    "dotdash", 
    # "dashdot",
    "dashed", 
]


bg_prior = noise_list[-1]

print(bg_prior)
noise_dist = bg_data[bg_prior+"/Signal"]

threshold = np.quantile(noise_dist, 0.95)

targeted_snr = 10
for key, items in inj_dict.items():
    
    p = figure(
        title=f"{key} output distribution", 
        x_axis_label="NN Model output value", 
        y_axis_label="Data count percentage", 
        width=1000, 
        height=1000
    )

    for i, mode in enumerate(noise_list):

        data = bg_data[mode+"/Signal"]

        bins = np.linspace(-25, 25, 100)
        hist, edges = np.histogram(data, density=True, bins=bins)

        p.line(
            (edges[:-1] + edges[1:])/2,
            hist, 
            line_color="black",
            # line_color=bg_colors[i],
            line_dash=line_patten[i],
            legend_label=f"{mode}",
            line_width=5
        )
        
        
    for j, item in enumerate(items):
        
        
        # print(item)
        data = sig_dis_data[item+f"_{targeted_snr:02d}/Signal"]

        bins = np.linspace(-25, 25, 100)
        hist, edges = np.histogram(data, density=True, bins=bins)
        
        p.line(
            (edges[:-1] + edges[1:])/2,
            hist, 
            # fill_color="black",
            line_color=inj_colors[j],
            legend_label=f"{item}",
            line_width=5
        )
        
    vline = Span(location=threshold, dimension='height', line_color='red', line_width=3)
    
    p.renderers.extend([vline])
    p.title.text_font_size = '20pt'
    p.title.text_font = "times"
    p.title.text_color = 'black'

    p.xaxis.axis_label_text_font = 'times'
    p.xaxis.axis_label_text_font_size = '20pt'
    p.xaxis.axis_label_text_color = 'black'
    p.yaxis.axis_label_text_font = 'times'
    p.yaxis.axis_label_text_font_size = '20pt'
    p.yaxis.axis_label_text_color = 'black'

    p.xaxis.major_label_text_font_size = '16pt'
    p.yaxis.major_label_text_font_size = '16pt'

    p.legend.click_policy="hide"
    p.legend.location = "top_left"
    p.legend.label_text_font_size = "15pt"
    
    local_path = the_louvre / f"snr_{targeted_snr:02d}_distribution"
    local_path.mkdir(parents=True, exist_ok=True)
    
    output_file(the_louvre / f"snr_{targeted_snr:02d}_distribution" / f"{key}_output_distribution.html")
    
    show(p)



for key, items in inj_dict.items():

    p = figure(
        title=f"{key}", 
        x_axis_label="SNR value", 
        y_axis_label="True positive rate", 
        width=800, 
        height=800
    )
    
    for j, item in enumerate(items):
        
        tprs = np.empty(len(snrs))*0
        for k, snr in enumerate(snrs):
            
            inj_dist = sig_dis_data[item+f"_{snr:02d}/Signal"]
            
            # print(item, len(inj_dist[inj_dist>threshold])/5000)
            tprs[k] = len(inj_dist[inj_dist>threshold])/5000

        p.line(
            snrs,
            tprs,
            legend_label=test_signals_dict[key][j],
            line_color=inj_colors[j],
            line_width=5
        )
        
        p.circle(
            snrs,
            tprs,
            color=inj_colors[j],
            size=10
        )
    
    
    p.title.text_font_size = '20pt'
    p.title.text_font = "times"
    p.title.text_color = 'black'

    p.xaxis.axis_label_text_font = 'times'
    p.xaxis.axis_label_text_font_size = '20pt'
    p.xaxis.axis_label_text_color = 'black'
    p.yaxis.axis_label_text_font = 'times'
    p.yaxis.axis_label_text_font_size = '20pt'
    p.yaxis.axis_label_text_color = 'black'

    p.xaxis.major_label_text_font_size = '16pt'
    p.yaxis.major_label_text_font_size = '16pt'

    p.legend.click_policy="hide"
    p.legend.location = "bottom_right"
    p.legend.label_text_font_size = "15pt"
    
    local_path = the_louvre / "TPRs_095"
    local_path.mkdir(parents=True, exist_ok=True)
    output_file(local_path / f"{key}.html")
    
    show(p)


def dis_trigger_pop(
    distance,
    nn_output,
    threshold,
    min_dis=0.1,
    max_dis=200,
    space_count=20,

):
    
    norm_population = np.zeros((2, space_count)) # Trigger%, pop_count
    
    bins = np.geomspace(min_dis, max_dis, space_count+1)
    
    bars = np.zeros(space_count)

    for i in range(space_count):
        bars[i] = np.sqrt(bins[i]*bins[i+1])
        

    for i in range(space_count):
        
        mask = np.logical_and(distance>bins[i], distance<bins[i+1])
        pop_counts = len(nn_output[mask])
        
        new_dis = nn_output[mask]
        pop_counts_alarm = (new_dis>threshold).sum()
        
        if pop_counts != 0:
            
            norm_population[0, i] = pop_counts_alarm/pop_counts
            norm_population[1, i] = pop_counts

    return bars, norm_population


dis_path = Path("/home/hongyin.chen/PlayGround/torch_tests")

for key, items in inj_dict.items():
    
    p = figure(
        title=f"{key}", 
        x_axis_label="Distance (1kpc)", 
        y_axis_label="True positive rate", 
        width=800, 
        height=800,
        x_axis_type="log",
        y_range=[0, 1.01]
    )
    p.add_layout(Legend(), 'right')
    print(key)
    new_dis = h5_thang(dis_path / f"dis_response_{key}.h5").h5_data()
    for j, item in enumerate(items):
    
        # print(""item)
        
        bars, norm_population = dis_trigger_pop(
            nn_output = new_dis[f"{item}/Output"],
            distance = new_dis[f"{item}/Distance"],
            threshold = np.quantile(noise_dist, 0.95),
            min_dis=1,
            max_dis=10,
            space_count=20,
        )


        p.line(
            bars,
            norm_population[0],
            legend_label=test_signals_dict[key][j],
            line_color=inj_colors[j],
            line_width=5
        )

        p.circle(
            bars,
            norm_population[0],
            color=inj_colors[j],
            size=10
        )

        
    p.title.text_font_size = '20pt'
    p.title.text_font = "times"
    p.title.text_color = 'black'

    p.xaxis.axis_label_text_font = 'times'
    p.xaxis.axis_label_text_font_size = '20pt'
    p.xaxis.axis_label_text_color = 'black'
    p.yaxis.axis_label_text_font = 'times'
    p.yaxis.axis_label_text_font_size = '20pt'
    p.yaxis.axis_label_text_color = 'black'

    p.xaxis.major_label_text_font_size = '16pt'
    p.yaxis.major_label_text_font_size = '16pt'

    p.legend.click_policy="hide"
    p.legend.location = "bottom_right"
    p.legend.label_text_font_size = "15pt"
    
    local_path = the_louvre / "Distance_095"
    local_path.mkdir(parents=True, exist_ok=True)
    output_file(local_path / f"{key}.html")
    
    show(p)