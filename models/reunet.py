import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import reunet_basicblock as B

import numpy as np
import torch.autograd as autograd

from scipy.io import loadmat

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2 as cv

 
def numpy_to_torch(a: np.ndarray):
    return torch.from_numpy(np.array(a)).float().permute(2, 0, 1)  # TL : force the cast to np array to avoid warning when call with jax array

def torch_to_numpy(a: torch.Tensor):
    return a.permute(1, 2, 0).numpy()

def convert_dict(base_dict, batch_sz):
    out_dict = []
    for b_elem in range(batch_sz):
        b_info = {}
        for k, v in base_dict.items():
            if isinstance(v, (list, torch.Tensor)):
                b_info[k] = v[b_elem]
        out_dict.append(b_info)

    return out_dict

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

def mosaic_tensor(image, mode='rggb'):
    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.shape
    if image.dim() == 3:
        image = image.unsqueeze(0)

    if mode == 'rggb':
        red = image[:, 0, 0::2, 0::2]
        green_red = image[:, 1, 0::2, 1::2]
        green_blue = image[:, 1, 1::2, 0::2]
        blue = image[:, 2, 1::2, 1::2]
        image = torch.stack((red, green_red, green_blue, blue), dim=1)
    elif mode == 'grbg':
        green_red = image[:, 1, 0::2, 0::2]
        red = image[:, 0, 0::2, 1::2]
        blue = image[:, 2, 0::2, 1::2]
        green_blue = image[:, 1, 1::2, 1::2]

        image = torch.stack((green_red, red, blue, green_blue), dim=1)

    if len(shape) == 3:
        return image.view((4, shape[-2] // 2, shape[-1] // 2))
    else:
        return image.view((-1, 4, shape[-2] // 2, shape[-1] // 2))

def demosaic_tensor(image):
    assert isinstance(image, torch.Tensor)
    image = image.clamp(0.0, 1.0) * 255

    if image.dim() == 4:
        num_images = image.dim()
        batch_input = True
    else:
        num_images = 1
        batch_input = False
        image = image.unsqueeze(0)

    # Generate single channel input for opencv
    im_sc = torch.zeros((num_images, image.shape[-2] * 2, image.shape[-1] * 2, 1))
    im_sc[:, ::2, ::2, 0] = image[:, 0, :, :]
    im_sc[:, ::2, 1::2, 0] = image[:, 1, :, :]
    im_sc[:, 1::2, ::2, 0] = image[:, 2, :, :]
    im_sc[:, 1::2, 1::2, 0] = image[:, 3, :, :]

    im_sc = im_sc.numpy().astype(np.uint8)

    out = []

    for im in im_sc:
        im_dem_np = cv.cvtColor(im, cv.COLOR_BAYER_BG2RGB_VNG)

        # Convert to torch image
        im_t = npimage_to_torch(im_dem_np, input_bgr=False)
        out.append(im_t)

    if batch_input:
        return torch.stack(out, dim=0)
    else:
        return out[0]

def single2uint(img):
    
    return np.uint8((img.clip(0, 1)*255.).round())


def unit162single(img):

    return np.float32(img/65535.)


def single2uint16(img):

    return np.uint8((img.clip(0, 1)*65535.).round())


# --------------------------------
# numpy(unit) <--->  tensor
# uint (HxWxn_channels (RGB) or G)
# --------------------------------


# convert uint (HxWxn_channels) to 4-dimensional torch tensor
def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)


# convert uint (HxWxn_channels) to 3-dimensional torch tensor
def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)


# convert torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())


# --------------------------------
# numpy(single) <--->  tensor
# single (HxWxn_channels (RGB) or G)
# --------------------------------


# convert single (HxWxn_channels) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


def single2tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1, 3).float().unsqueeze(0)


def single42tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1, 3).float()

# convert single (HxWxn_channels) to 3-dimensional torch tensor
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


"""
# --------------------------------------------
# (1) Prior module; ResUNet: act as a non-blind denoiser
# x_k = P(z_k, beta_k)
# --------------------------------------------
"""

class ResUNet(nn.Module):
    def __init__(self, in_nc=12, out_nc=1, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, input):
        x = input[0]
        mask = input[1]
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return torch.multiply(x, mask)

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from datasets.unet_dataset import UNetDataset
    import matplotlib.pyplot as plt
    from scripts.submit import submit
    from scripts.validate import validate
    from scripts.train import train
    from utils.image import torch_to_numpy
    import torchvision.transforms as transforms

    model = ResUNet()
    PATH = "models_ckpt\_epoch_94_iter_30.pth"
    model.load_state_dict(torch.load(PATH))
    val_dir = "challenge_data/validation/validation"
    train_dir = "challenge_data/train/train"
    test_dir = "challenge_data/test/test"
    
    transform_val = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(size=(110,80))])
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(size=(110,80)),
                            transforms.RandomAffine(degrees=(0, 0), translate=(0., 0.), scale=(0.8, 1.2))])
    test_dataset = UNetDataset(test_dir, transforms=transform_val, test=True)
    train_dataset = UNetDataset(train_dir, transforms=transform)
    val_dataset = UNetDataset(val_dir, transforms=transform_val)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                          shuffle=True)
    # for elem in iter(train_dataset):
    #     for el, label in zip(elem['input'][0], elem["label"]):
    #         plt.figure()
    #         plt.subplot(1,2,1)
    #         plt.imshow(torch_to_numpy(el.unsqueeze(0).detach().cpu()))
    #         plt.subplot(1,2,2)
    #         plt.imshow(torch_to_numpy(label.unsqueeze(0).detach().cpu()))
    #         plt.show()
    #         break
    # train(model.cuda(), train_dataset, val_dataset, batch_size=32)
    # print(validate(model.cuda(), loader))
    # for sample in loader:
    #     # print(l[0][0].shape)
    #     # print(l[1].shape)
    #     l = sample["input"]
    #     print(l[0][0].shape)
    #     print(sample["label"].shape)
    #     output = model(l)
    #     # output = 85 * (output/output.max())
    #     print(output.shape)
    #     plt.figure()
    #     plt.subplot(1,3,1)
    #     plt.imshow(torch_to_numpy(output[0]))
    #     plt.subplot(1,3,2)
    #     plt.imshow(torch_to_numpy(l[0][0][0:1]))
    #     plt.subplot(1,3,3)
    #     plt.imshow(torch_to_numpy(sample["label"][0]))
    #     plt.show()
    
    validate(model.cuda(), val_dataset)

    # test_dir = "challenge_data/test/test"
    # test_dataset = ToyDataset(test_dir, test=True)
    # model.test = True

    submit(model=model.cuda(), 
           dataset=loader,
           name="resunet3")