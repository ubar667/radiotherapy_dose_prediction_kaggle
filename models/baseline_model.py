import numpy as np
import os
from scipy.ndimage.morphology import binary_dilation, distance_transform_edt, distance_transform_bf


class BaselineModel():
    def __init__(self, n_dilation=20, max_value=75):
        self.n_dilation = n_dilation
        self.max_value = max_value
        self.test = True

    def __call__(self, sample):
        possible_dose_mask = sample["possible_dose_mask"]
        all_mask = sample["all_good_structure_mask"]
        if all_mask.sum() ==0.:
            all_mask = sample["all_structure_mask"]
        if all_mask.sum() ==0.:
            return distance_transform_edt(possible_dose_mask)

        output = distance_transform_edt(
            binary_dilation(all_mask, iterations=self.n_dilation)
            )
        output /= output.max()
        output = output * self.max_value
        output[all_mask == 1] = self.max_value
        output = np.multiply(output, possible_dose_mask)

        return output

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from datasets.toy_dataset import ToyDataset
    import matplotlib.pyplot as plt
    from scripts.submit import submit
    from scripts.validate import validate

    toy_model = BaselineModel()
    val_dir = "challenge_data/validation/validation"
    val_dataset = ToyDataset(val_dir)
    validate(toy_model, val_dataset)

    test_dir = "challenge_data/test/test"
    test_dataset = ToyDataset(test_dir, test=True)
    toy_model.test = True

    submit(model=toy_model, 
           dataset=test_dataset,
           name="toy1")