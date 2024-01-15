import torch

import numpy as np


class Validator:
    
    def __init__(
        self,
        station_bg,
        glitch,
        signal,
        validation_fraction,
        injection_control, # Distance, SNR, time_shift ### We might need another function here (ML4GW)
        batch_size: int,
        validaton_type: [list, dict], # bg, glitch, bg+signal, glitch+signal
        metric_method: [list, dict],  # Exculsion-socre, AUC-score
        stream: bool = False # If true, stream in data by a sliding window on the active segments then save infos of i/o
        ):
        
        self.three = 3
        
    def __call__(self):
        
        pass



# Read CCSN h5 files
# Label each CCSN them by name

# Sample BG, Glitch, CCSN
# Injection rescalling
# Whiten, Cropping

####################
### Model Stream ###
####################

# This function should interact with validation scheme
### The validation scheme should includes 
# (1). Recall at var(??)% of glitch exclusion
# (2). 
# def forward ...
#   return valdation result
        