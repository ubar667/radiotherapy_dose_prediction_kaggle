from datasets.datasets import MedicalDataset
import numpy as np


class UNetDataset(MedicalDataset):
    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        if self.test:
            mask = sample["possible_dose_mask"]
            x = np.concatenate([ sample["ct"][None, ...]/1000., sample["possible_dose_mask"][None, ...], sample["structure_masks"]]).transpose(1,2,0)
            if self.transforms is not None:
                x = self.transforms(x)
                mask = self.transforms(mask)   
            
            sample["input"] = [x.float(), mask.float()]
            return sample 

        dose_norm = sample["dose"]
        dose_norm = np.multiply(dose_norm, sample["possible_dose_mask"])
        x = np.concatenate([ sample["ct"][None, ...]/1000., sample["possible_dose_mask"][None, ...], 
                            sample["structure_masks"],dose_norm[None, ...]]).transpose(1,2,0)

        mask = sample["possible_dose_mask"]

        
        if self.transforms is not None:
            x = self.transforms(x)
            label = x[12:13, ...]
            x = x[0:12, ...]
            mask = self.transforms(mask)

            sample["input"] = [x.float(), mask.float()]
            sample["label"] = label
        
        return sample
