import torch


def forged_dataloader(
    inputs: list,
    targets: list,
    batch_size,
    pin_memory=True
):

    dataset = torch.utils.data.TensorDataset(
        torch.cat(inputs).to("cuda"), 
        torch.cat(targets).view(-1, 1).to(torch.float).to("cuda")
    )
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_time_sampling(
    background_sampler,
    signal_sampler,
    max_distance,
    batch_size,
    steps_per_epoch,
    iteration,
    noise_glitch_dist,
    signal_glitch_dist,
    sample_factor=1/2,
    choice_mask = [0, 1, 2, 3],
    glitch_offset = 0.9
):
    

    glitch_signal, glitch_target = background_sampler(
        batch_size = batch_size,
        steps_per_epoch = steps_per_epoch,
        sample_factor = sample_factor,
        glitch_dist = noise_glitch_dist,
        iteration=iteration,
        mode="background",
        choice_mask = choice_mask,
        glitch_offset = glitch_offset,
        target_value = 0
    )

    injection_siganl, injection_target = background_sampler(
        batch_size = batch_size,
        steps_per_epoch = steps_per_epoch,
        sample_factor = sample_factor,
        glitch_dist = signal_glitch_dist,
        iteration=iteration,
        mode="glitch",
        choice_mask = choice_mask,
        glitch_offset = glitch_offset,
        target_value = 1
    )


    injected_siganl = signal_sampler(
        background=injection_siganl,
        iteration=iteration,
        max_distance=max_distance
    )


    training_loader = forged_dataloader(
        inputs = [glitch_signal, injected_siganl],
        targets = [glitch_target, injection_target],
        batch_size = batch_size
    )

    return training_loader