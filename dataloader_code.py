from torch.utils.data import Dataset
import h5py
import numpy as np
import torch


class MyECGDataset(Dataset):
    def __init__(self, path):
        
        # open pointer to h5 file
        self.f = h5py.File(path, "r")

        # length of dataset
        self.dset_length = len(self.f["x_ecg_test"])
        # load ids, labels, age, sex immediately since those are not too large
        self.ids = torch.tensor(self.f["id_exam"])
        self.register_num = torch.tensor(self.f["register_num"])
        self.labels = torch.tensor(self.f["y_test"]).long()
        self.age = torch.tensor(self.f["x_age_test"])
        self.sex = torch.tensor(self.f["x_is_male_test"]).long()

        age_mean, age_std = 62.60858, 19.514
        self.normalize_age(age_mean, age_std)

    def get_num_leads_outputs(self):
        # get number of leads / outputs
        try:
            num_leads = self.f["x_ecg_train_nodup"][0].shape[0]
            num_outputs = len(np.unique(self.f["y_train_nodup"]))
        except:
            # default values
            num_leads = 8
            num_outputs = 3
        return num_leads, num_outputs

    def normalize_age(self, mean, std):
        # normalize age
        self.age_normalized = (self.age - mean) / std

    def __len__(self):
        return self.dset_length

    def __getitem__(self, item):
        """
        return: traces, labels, ids, regs, age, age_norm and sex
        """
        traces = self.f["x_ecg_test"][item].astype(np.float32)
        labels = self.labels[item]
        ids = self.ids[item]
        regs = self.register_num[item]
        age = self.age[item]
        age_norm = self.age_normalized[item]
        sex = self.sex[item]
        
        return traces, labels, ids, regs, age, age_norm, sex
