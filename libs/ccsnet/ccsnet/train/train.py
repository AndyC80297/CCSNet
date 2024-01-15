import toml
import torch
import logging

from torch.profiler import schedule, tensorboard_trace_handler
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

from ccsnet.arch import WaveNet
from ml4gw.transforms import SnrRescaler

ARGUMENTS_FILE = "/home/hongyin.chen/anti_gravity/CCSNet/apps/train/trainer/arguments.toml"

ccsnet_arguments = toml.load(ARGUMENTS_FILE)

def one_loop_training(
    train_data,
    validation_scheme,
    model,
    criterion,
    opt,
    lr_scheduler,
    iteration,
    device
):
    
    t_cost = 0
    
    for j, (x, y) in enumerate(tqdm(train_data)):
    # for j, (x, y) in enumerate(train_data):
        x = x[:, :, :4096].to(device)
        y = y.to(device)

        p_value = model(x)

        # cost = criterion(p_value, torch.argmax(y, dim = 1))
        cost = criterion(p_value, y)
        opt.zero_grad()
        cost.backward()
        opt.step()
        
        # print(cost.item())


def Tachyon(
    architecture: Callable,
    # Plz provide in memeory control for the data
    train_data,  # Processed data
    validation_scheme,  # Stream in diffent validation method process data when called
    batch_size = ccsnet_arguments["batch_size"],
    max_iteration = ccsnet_arguments["max_iteration"],
    num_ifo = len(ccsnet_arguments["ifos"]),
    pretrained_model = None,
    weight_decay = 1e-5,
    learning_rate = 0.01,
    device: str = "cuda",
    outdir: Path = Path("/home/hongyin.chen/Outputs/CCSNet_Out_dir/pseudo_output"),
    softmax_output_layer:bool = False,
    enhancer = {
        "vram": True,
        "speed": True,
        "precision": True,
        "profile": False
        }
):
    
    device = device or "cpu"
    
    model = architecture(num_ifo)
    model.to(device)
    
    logging.info
    
    # First Style
    if softmax_output_layer:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.functional.binary_cross_entropy_with_logits
    # Second Style
    

    opt = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay # 1e-5
    )
    
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=learning_rate, # max_lr
        epochs=max_iteration,
        steps_per_epoch=len(train_data),
        anneal_strategy="cos",
    )
    
    # Add a enhacer that helps to prelocated vram 
    torch.backends.cudnn.benchmark = enhancer["speed"]
    
    if enhancer["precision"] and device.startswith("cuda"):
        
        scaler = torch.cuda.amp.GradScaler()
        
    logging.info("=========================================")
    logging.info("|| Tachyon enhanced!!! Starts training ||")
    logging.info("=========================================")
    
    for iteration in range(max_iteration):
        if iteration == 0 and enhancer["profile"]:
            profiler = torch.profiler.profile(
                schedule=schedule(wait=0, warmup=1, active=10),
                on_trace_ready=tensorboard_trace_handler(outdir / "profile")
            )
            profiler.start()
        else:
            profiler = None
            
            
        logging.info(f"=== Epoch {iteration + 1}/{max_iteration} ===")
        # Add iteration caching
        early_stopping = one_loop_training(
            train_data,
            validation_scheme,
            model,
            criterion,
            opt,
            lr_scheduler,
            iteration,
            device
        )
        
        if early_stopping:
            
            break
        
