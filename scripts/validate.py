import numpy as np
from tqdm import tqdm
from utils.image import torch_to_numpy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def validate(model, dataset, pad=nn.ZeroPad2d((24,24, 9, 9))):
    """Compute loss over validation dataset

    Args:
        model (nn.Module): model to validate
        dataset (MedicalDataset): validation dataset
        pad (nn.ZeroPad, optional): padding to recover a size of 128x128. Defaults to nn.ZeroPad2d((24,24, 9, 9)).

    Returns:
        float: validation error
    """
    losses = []
    model.eval()
    valcriterion = nn.L1Loss()
    valloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    with torch.no_grad():
        for sample in tqdm(valloader):
            val_in = [sample["input"][0].cuda(), sample["input"][1].cuda()]
            out = model(val_in)
            if pad is not None:
                out = pad(out)
            out = torch.multiply(out, sample["possible_dose_mask"].cuda().unsqueeze(1))
            losses.append(valcriterion(out, sample["dose"].cuda().unsqueeze(1)).item())
    return sum(losses)/len(losses)