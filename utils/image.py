import numpy as np
import torch
import cv2 as cv

def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(a).float().permute(2, 0, 1)


def torch_to_numpy(a: torch.Tensor):
    return a.detach().cpu().permute(1, 2, 0).numpy()


def torch_to_npimage(a: torch.Tensor, unnormalize=True):
    a_np = torch_to_numpy(a)

    if unnormalize:
        a_np = a_np * 255
    a_np = a_np.astype(np.uint8)
    return cv.cvtColor(a_np, cv.COLOR_RGB2BGR)

def npimage_to_torch(a, normalize=True, input_bgr=True):
    if input_bgr:
        a = cv.cvtColor(a, cv.COLOR_BGR2RGB)
    a_t = numpy_to_torch(a)

    if normalize:
        a_t = a_t / 255.0

    return a_t