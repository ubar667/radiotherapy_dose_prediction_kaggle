from datasets import MedicalDataset
from tqdm import tqdm
class ToyDataset(MedicalDataset):
    def __getitem__(self, idx):
        return self.get_sample(idx)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage.morphology import binary_dilation, distance_transform_edt
    from visualisation import show_organs, show_dose_organs, show_sample, show_array
    train_dir = "challenge_data/validation/validation"

    train_dataset = ToyDataset(train_dir)
    sample = next(iter(train_dataset))
    mean_ct = 0
    std_ct = 0
    for sample in tqdm(train_dataset):
        ct = sample["ct"]
        possible_dose_mask = sample["possible_dose_mask"]
        structure_masks = sample["structure_masks"]
        dose = sample["dose"]
        all_mask = sample["all_structure_mask"]
        mean_ct += structure_masks.mean()
        std_ct += structure_masks.std()
        # output = distance_transform_edt(binary_dilation(all_mask, iterations=20))
    print(mean_ct/len(train_dataset))
    print(std_ct/len(train_dataset))
        # output /= output.max()
        # output[all_mask == 1] = 1 
        # output = np.multiply(output, possible_dose_mask)
        # # output = 80*output
        # show_array(output)
        # # output += output.mean() * np.multiply(1-all_mask, possible_dose_mask)
        
        # keys = ["ct", "possible_dose_mask", "dose"]
        # structure_masks = sum([0.1*i*organ for i, organ in enumerate(sample["structure_masks"])])
        
        # plt.figure()
        # for i, key in enumerate(keys):
        #     plt.subplot(1, 5, i+1)
        #     plt.imshow(sample[key])
        # plt.subplot(1, 5, 4)
        # plt.imshow(structure_masks)
        # plt.subplot(1, 5, 5)
        # plt.imshow(output)
        # plt.show()
        
        # show_array(output)
        # show_organs(sample)
        # show_dose_organs(sample)
        # show_sample(sample)
        # print(sample["organ_dose"].shape)
        # break
        # print(dose[dose != 0.].min())
        # print(dose[dose != 0.].max())
        # print(dose[dose != 0.].mean())
        # print(dose[dose != 0.].std())
        
