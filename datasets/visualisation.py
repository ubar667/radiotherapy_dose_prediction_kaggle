import matplotlib.pyplot as plt
import numpy as np

def show_organs(sample):
    structure_masks = sample["structure_masks"]
    plt.figure()
    for i in range(10):
        plt.subplot(5, 5, i+1)
        plt.imshow(structure_masks[i, ...])
    plt.show()

def show_dose_organs(sample):
    organ_dose = sample["organ_dose"]
    plt.figure()
    for i in range(10):
        plt.subplot(5, 5, i+1)
        plt.imshow(organ_dose[i, ...])
    plt.show()

def show_sample(sample):
    keys = ["ct", "possible_dose_mask", "dose"]
    structure_masks = sum([0.1*i*organ for i, organ in enumerate(sample["structure_masks"])])
    plt.figure()
    for i, key in enumerate(keys):
        plt.subplot(1, 4, i+1)
        plt.imshow(sample[key])
    plt.subplot(1, 4, 4)
    plt.imshow(structure_masks)
    plt.show()

def show_array(arr):
    plt.figure()
    plt.imshow(arr)
    plt.show()
    
