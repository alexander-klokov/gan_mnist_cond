import torch
from torch.utils.data import Dataset

import pandas
import matplotlib.pyplot as plt

class MnistDataset(Dataset):

    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, low_memory = False)
        pass

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0]

        target = torch.zeros((10))
        target[int(label)] = 1.0

        image_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values) / 255.0

        return label, image_values, target

    def plot_image(self, index):
        label = self.data_df.iloc[index, 0]
        arr = self.data_df.iloc[index, 1:].values.reshape(28,28).astype(float)

        plt.figure(figsize = (15,10))
        plt.title(label)
        plt.imshow(arr, interpolation="none", cmap='gray', vmin=0, vmax=255)

        plt.show()
        pass

    pass
