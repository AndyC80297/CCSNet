import toml
import torch
import logging

from torch.profiler import schedule, tensorboard_trace_handler
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

from ccsnet.arch import WaveNet
from ccsnet.train import train_time_sampling
from ml4gw.transforms import SnrRescaler

# ARGUMENTS_FILE = "/home/hongyin.chen/anti_gravity/CCSNet/apps/train/trainer/arguments.toml"

# ccsnet_arguments = toml.load(ARGUMENTS_FILE)

def one_loop_training(
    background_sampler,
    train_data,
    validation_scheme,
    max_distance,
    whiten_model,
    psds,
    model,
    criterion,
    optimizer,
    lr_scheduler,
    scaler,
    profiler,
    iteration,
    batch_size,
    steps_per_epoch,
    outdir,
    device
):
    model.train()
    t_cost = 0
    samples_seen = 0
    # psds.to(device)
    
    # print(psds.get_device())
    # print(x.get_device())
    for j, (x, y) in enumerate(tqdm(train_data)):
        
        optimizer.zero_grad(
            # set_to_none=True
        )
        
        x = whiten_model(
            x, 
            psds
        )

        # p_value = model(x)

        # # cost = criterion(p_value, torch.argmax(y, dim = 1))
        # cost = criterion(p_value, y)

        # cost.backward()
        # opt.step()

        with torch.autocast("cuda", enabled=scaler is not None):
            predictions = model(x)
            loss = criterion(predictions, y)
        t_cost += loss.item()
        samples_seen += len(x)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if profiler is not None:
            profiler.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        t_cost += loss.item()
        
    # average_cost = t_cost/(steps_per_epoch)
    average_cost = t_cost/(j+1)
    logging.info("")
    logging.info(f"    ============================")
    logging.info(f"    === Training cost {average_cost:.4f} ===")
    logging.info(f"    === Training cost {loss.item():.4f} ===")
    logging.info(f"    ============================")
    logging.info("")
    
    ###### Need to update distance
    model.eval()
    distance = validation_scheme(
        loss=average_cost,
        back_ground_display=background_sampler,
        # batch_size,
        model=model,
        criterion=criterion,
        whiten_model=whiten_model,
        psds=psds,
        iteration=iteration,
        max_distance=max_distance,
        # outdir=outdir,
        device=device
    )


    if iteration % 5 ==0:

        torch.save(model.state_dict(), outdir/f"models/Iter{iteration:03d}")


    return distance

def Tachyon(
    architecture: Callable,
    # Plz provide in memeory control for the data
    background_sampler, 
    signal_sampler,
    max_distance,
    noise_glitch_dist,
    signal_glitch_dist,
    validation_scheme,
    whiten_model,
    psds,
    batch_size,
    steps_per_epoch,
    max_iteration,
    num_ifo,
    pretrained_model = None,
    weight_decay = 1e-5,
    learning_rate = 0.01,
    device: str = "cpu",
    outdir: Path = None,
    softmax_output_layer:bool = False,
    enhancer = {
        "vram": True,
        "speed": True,
        "precision": True,
        "profile": False
        }
):
    
    device = device or "cpu"
    model_dir = outdir / f"models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
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
        steps_per_epoch=steps_per_epoch,
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
        
        
        train_data = train_time_sampling(
            background_sampler,
            signal_sampler,
            max_distance,
            batch_size,
            steps_per_epoch,
            iteration = iteration, 
            sample_factor=1/2,
            noise_glitch_dist = noise_glitch_dist,
            signal_glitch_dist = signal_glitch_dist,
            choice_mask = [0, 1, 2, 3],
            glitch_offset = 0.9
        )
        
        
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
        max_distance = one_loop_training(
            background_sampler,
            train_data,
            validation_scheme,
            max_distance,
            whiten_model,
            psds,
            model,
            criterion,
            opt,
            lr_scheduler,
            scaler,
            profiler,
            iteration,
            batch_size,
            steps_per_epoch,
            outdir,
            device
        )
        
        # if early_stopping:
            
        #     break
        
