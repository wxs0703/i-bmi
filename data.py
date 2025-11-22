import pandas as pd
import torch
from torch.utils.data import Dataset

class IBMIDataset(Dataset):
    def __init__(self, csv_file):
        '''
        Args:
            csv_file (string): Path to the csv file with data.
        '''
        self.data_frame = pd.read_csv(csv_file)
        self.label_cols = ['Told_High_Cholesterol',
                           'Diagnosed_Diabetes',
                           'Diagnosed_Hypertension',
                           'Diagnosed_Thyroid_Problem',
                           'CVD_Diagnosed']
        self.features = self.data_frame.drop(columns=self.label_cols + ['User_ID'])
        self.labels = self.data_frame[self.label_cols]

    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature_sample = self.features.iloc[idx].values.astype('float32')
        label_sample = self.labels.iloc[idx].values.astype('float32')

        sample = {'features': torch.tensor(feature_sample),
                  'labels': torch.tensor(label_sample)}

        return sample