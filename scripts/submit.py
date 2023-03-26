import numpy as np
import os
from tqdm import tqdm
from utils.data import file_compress
from utils.image import torch_to_numpy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def submit(model, dataset, dir="submissions", name="toy", pad=nn.ZeroPad2d((24,24, 9, 9))):
    file_paths = []
    dataset.test = True
    print(f"*** save .npy samples")
    err = 0
    max_l = []
    min_l = []
    
    m = []
    std = []
    for sample in tqdm(dataset):
        test_in = [sample["input"][0].cuda(), sample["input"][1].cuda()]
        out = model(test_in)
        if pad is not None:
            out = m(out)
        out = torch.nn.ReLU()(out)
        out = torch.multiply(out, sample["possible_dose_mask"].cuda().unsqueeze(1))
        out = torch_to_numpy(out[0].cpu().detach()).squeeze()

        file_path = os.path.join(dir, sample['sample_name'][0]+'.npy')
        sub_path = os.path.join("submissions", "submission", sample['sample_name'][0]+'.npy')
        im = np.load(sub_path)#[...,None]
        
        ct = torch_to_numpy(sample["ct"])
        ct = ct[ct>0.]
        
        
        np.save(file_path, out)
        
        file_paths.append(file_path)

    out_zip = os.path.join(dir, name)
    file_compress(file_paths, out_zip)
    
    print(f' *** removing files ')
    for file in tqdm(file_paths):
        os.remove(file)
